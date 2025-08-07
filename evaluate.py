#!/usr/bin/env python
# coding: utf-8

import os
import glob
import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from config_trAISformer import Config
from encoder import Encoder
import models, datasets, utils, trainers
from torch.utils.data import DataLoader

def evaluate_checkpoint(ckpt_path, cf, test_loader, device):
    # 모델 초기화
    encoder = Encoder(cf).to(device)
    model   = models.TrAISformer(cf, partition_model=None).to(device)

    # 체크포인트 로드 (weights_only=True 사용 가능 시)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    model.load_state_dict(ckpt['decoder_state_dict'])
    encoder.eval()
    model.eval()

    # 평가 설정
    init_seqlen = cf.init_seqlen
    max_seqlen  = init_seqlen + 6 * 4  # 4시간 예측
    v_ranges    = torch.tensor([model.lat_range, model.lon_range, 0, 0]).to(device)
    v_roi_min   = torch.tensor([model.lat_min, model.lon_min, 0, 0]).to(device)

    all_errors = []

    with torch.no_grad():
        for (enc_in, seqs, _, masks, *_) in tqdm(test_loader, desc=os.path.basename(ckpt_path)):
            B = seqs.size(0)
            seqs = seqs.to(device)
            enc_in      = enc_in.to(device)                             # (B, T)
            seqs_init   = seqs[:, :init_seqlen, :].to(device)           # (B, init_seqlen, 4)
            encoder_out = encoder(enc_in).to(device)                               # (B, T, d)
            masks_batch = masks[:, :max_seqlen].to(device)              # (B, max_seqlen)

            # N번 샘플링 → best-of-N
            error_ens = torch.zeros((B, max_seqlen - init_seqlen, cf.n_samples), device=device)
            for i in range(cf.n_samples):
                preds = trainers.sample(
                    model,
                    encoder_out,
                    seqs_init,
                    max_seqlen - init_seqlen,
                    temperature=1.0,
                    sample=True,
                    sample_mode=cf.sample_mode,
                    r_vicinity=cf.r_vicinity,
                    top_k=cf.top_k
                )
                inp_coords  = (seqs[:, :max_seqlen, :] * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds               * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(inp_coords, pred_coords) * masks_batch
                error_ens[:, :, i] = d[:, init_seqlen:]

            best_err = error_ens.min(dim=-1).values  # (B, L)
            all_errors.append(best_err.cpu().numpy())

    all_errors  = np.concatenate(all_errors, axis=0)          # (total_samples, L)
    mean_errors = np.nanmean(all_errors, axis=0)             # (L,)

    return mean_errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True, help='체크포인트 디렉터리')
    parser.add_argument('--out_dir',  required=True, help='메트릭 저장 디렉터리')
    args = parser.parse_args()

    cf     = Config()
    device = cf.device

    # 테스트 데이터 로드 및 DataLoader 생성
    with open(os.path.join(cf.datadir, cf.testset_name), 'rb') as f:
        raw = pickle.load(f)
    # 기존 전처리 로직을 재사용하세요!
    test_dataset = datasets.AISDataset(
        raw,
        max_seqlen=cf.max_seqlen + 1,
        device=device,
        density_map_path=cf.datadir + cf.density_map_path,
        lat_min=cf.lat_min, lat_max=cf.lat_max,
        lon_min=cf.lon_min, lon_max=cf.lon_max,
        density_size=cf.density_size
    )
    test_loader = DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=False)

    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    for ckpt_path in sorted(glob.glob(os.path.join(args.ckpt_dir, '*.pt'))):
        mean_err = evaluate_checkpoint(ckpt_path, cf, test_loader, device)

        name = os.path.splitext(os.path.basename(ckpt_path))[0]
        np.save(os.path.join(args.out_dir, f"{name}_errors.npy"), mean_err)

        summary.append({
            'model': name,
            '1h': mean_err[6].item(),
            '2h': mean_err[12].item(),
            '3h': mean_err[18].item(),
        })

    df = pd.DataFrame(summary).set_index('model')
    df.to_csv(os.path.join(args.out_dir, 'summary.csv'))
    print("Done. Metrics saved under", args.out_dir)

if __name__ == "__main__":
    main()