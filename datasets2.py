# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """
    def __init__(self, 
                 l_data, 
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 density_map_path=None, # Path to the pre-computed density map
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None, # ROI for density map
                 density_size=None, pad_width=2):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """    
            
        self.max_seqlen = max_seqlen
        self.device = device
        
        self.l_data = l_data 
        self.pad_width = pad_width
        self.radius = 2
        
        # Load density map if provided
        self.density_map = None
        if density_map_path:
            with open(density_map_path, 'rb') as f:
                self.density_map = pickle.load(f)
                self.dm_pad = np.pad(self.density_map, pad_width=self.pad_width, mode="constant", constant_values=0) # zero padding 
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.lon_min = lon_min
            self.lon_max = lon_max
            self.density_size = density_size
            
            if self.density_map is not None:
                self.density_map_lat_res = (lat_max - lat_min) / self.density_map.shape[0]
                self.density_map_lon_res = (lon_max - lon_min) / self.density_map.shape[1]
                self.max_density_value = self.density_map.max()
                self.min_density_value = self.density_map.min()


    def _latlon_to_idx(self, lat, lon):
        # 패딩 전 map에서 lat/lon → int 인덱스
        i = int((lat - self.lat_min) / self.density_map_lat_res)
        j = int((lon - self.lon_min) / self.density_map_lon_res)
        H, W = self.density_map.shape
        # radius 까지 슬라이스 가능하도록 clamp
        max_i = H - 1 - self.radius
        max_j = W - 1 - self.radius
        i = max(0, min(i, max_i))
        j = max(0, min(j, max_j))
        return i, j

    def get_local_density_window(self, lat, lon, radius=None):
        # radius 기본값 처리
        if radius is None:
            radius = self.radius

        # 1) 실제 위경도 → 패딩 전 인덱스
        i, j = self._latlon_to_idx(lat, lon)
        # 2) 패딩 위치로 이동
        i_p, j_p = i + self.pad_width, j + self.pad_width
        # 3) 슬라이스 경계 계산 (정수)
        r = int(radius)
        i0, i1 = i_p - r, i_p + r + 1
        j0, j1 = j_p - r, j_p + r + 1

        # 4) 안전하게 맵 범위 내로 clamp (optional)
        H_p, W_p = self.dm_pad.shape
        i0 = max(0, i0); i1 = min(H_p, i1)
        j0 = max(0, j0); j1 = min(W_p, j1)

        return self.dm_pad[i0:i1, j0:j1]

    def __len__(self):
        return len(self.l_data)
        
    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
#         m_v[m_v==1] = 0.9999
        m_v[m_v>0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        
        # Prepare density sequence for encoder input
        density_seq = np.zeros((self.max_seqlen, self.radius*2+1, self.radius*2+1), dtype=np.float32) # (maxlen, 3, 3)
        for i in range(seqlen):
            lat, lon = m_v[i,0], m_v[i,1] # Use original lat/lon from trajectory for density lookup
            # Convert normalized lat/lon back to original range for density lookup
            original_lat = lat * (self.lat_max - self.lat_min) + self.lat_min if self.lat_max and self.lat_min else lat
            original_lon = lon * (self.lon_max - self.lon_min) + self.lon_min if self.lon_max and self.lon_min else lon
            
            window = self.get_local_density_window(original_lat, original_lon, self.radius)
            

            density_seq[i] = window 
 
            

        encoder_input_density = torch.tensor(density_seq, dtype=torch.float32)
        
        # Decoder input: trajectory features (lat, lon, sog, cog) up to seqlen-1
        decoder_input_seq = torch.tensor(m_v, dtype=torch.float32)[:seqlen, :]
        decoder_input_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        decoder_input_seq_padded[:seqlen, :] = decoder_input_seq

        # Decoder target: next token prediction. If input is x_0...x_T-1, target is x_1...x_T
        decoder_target_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        if seqlen > 1:
            decoder_target_seq_padded[:seqlen-1, :] = decoder_input_seq[1:seqlen, :]

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen_tensor = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)

        return encoder_input_density, decoder_input_seq_padded, decoder_target_seq_padded, mask, seqlen_tensor, mmsi, time_start

    
class AISDataset_grad(Dataset):
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """
    def __init__(self,
                 l_data,
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu"),
                 density_map_path=None, # Path to the pre-computed density map
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None, # ROI for density map
                 density_size=None, pad_width=2 ):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """

        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data
        self.pad_width = pad_width
        self.radius = 2

        # Load density map if provided
        self.density_map = None
        if density_map_path:
            with open(density_map_path, 'rb') as f:
                self.density_map = pickle.load(f)
                self.dm_pad = np.pad(self.density_map, pad_width=self.pad_width, mode="constant", constant_values=0) # zero padding 
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.lon_min = lon_min
            self.lon_max = lon_max
            self.density_size = density_size

            if self.density_map is not None:
                self.density_map_lat_res = (lat_max - lat_min) / self.density_map.shape[0]
                self.density_map_lon_res = (lon_max - lon_min) / self.density_map.shape[1]
                self.max_density_value = self.density_map.max()
                self.min_density_value = self.density_map.min()


    def _latlon_to_idx(self, lat, lon):
        # 패딩 전 map에서 lat/lon → int 인덱스
        i = int((lat - self.lat_min) / self.density_map_lat_res)
        j = int((lon - self.lon_min) / self.density_map_lon_res)
        H, W = self.density_map.shape
        # radius 까지 슬라이스 가능하도록 clamp
        max_i = H - 1 - self.radius
        max_j = W - 1 - self.radius
        i = max(0, min(i, max_i))
        j = max(0, min(j, max_j))
        return i, j

    def get_local_density_window(self, lat, lon, radius=None):
        # radius 기본값 처리
        if radius is None:
            radius = self.radius

        # 1) 실제 위경도 → 패딩 전 인덱스
        i, j = self._latlon_to_idx(lat, lon)
        # 2) 패딩 위치로 이동
        i_p, j_p = i + self.pad_width, j + self.pad_width
        # 3) 슬라이스 경계 계산 (정수)
        r = int(radius)
        i0, i1 = i_p - r, i_p + r + 1
        j0, j1 = j_p - r, j_p + r + 1

        # 4) 안전하게 맵 범위 내로 clamp (optional)
        H_p, W_p = self.dm_pad.shape
        i0 = max(0, i0); i1 = min(H_p, i1)
        j0 = max(0, j0); j1 = min(W_p, j1)

        return self.dm_pad[i0:i1, j0:j1]


    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            encoder_input_density: Tensor of (max_seqlen, 1) containing only density for encoder.
            decoder_input_seq: Tensor of (max_seqlen, 4) [lat, lon, sog, cog] for decoder input.
            decoder_target_seq: Tensor of (max_seqlen, 4) [lat, lon, sog, cog] for decoder target.
            mask: Tensor of (max_seqlen). mask[i] = 0.0 if x[i] is a padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4] # lat, lon, sog, cog
        m_v[m_v==1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)

        # Prepare density sequence for encoder input
        density_seq = np.zeros((self.max_seqlen, self.radius*2+1, self.radius*2+1), dtype=np.float32) # (maxlen, 3, 3)
        for i in range(seqlen):
            lat, lon = m_v[i,0], m_v[i,1] # Use original lat/lon from trajectory for density lookup
            # Convert normalized lat/lon back to original range for density lookup
            original_lat = lat * (self.lat_max - self.lat_min) + self.lat_min if self.lat_max and self.lat_min else lat
            original_lon = lon * (self.lon_max - self.lon_min) + self.lon_min if self.lon_max and self.lon_min else lon
            
            window = self.get_local_density_window(original_lat, original_lon, self.radius)

            density_seq[i] = window 


        encoder_input_density = torch.tensor(density_seq, dtype=torch.float32)

        # Decoder input: trajectory features (lat, lon, sog, cog)
        # The `m_v` here already contains the dlat/dlon in columns 2 and 3 if this is AISDataset_grad.
        # So `decoder_input_seq` will be `[lat, lon, dlat, dlon]`
        decoder_input_seq = torch.tensor(m_v, dtype=torch.float32)[:seqlen, :]
        decoder_input_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        decoder_input_seq_padded[:seqlen, :] = decoder_input_seq

        # Decoder target: next token prediction.
        decoder_target_seq_padded = torch.zeros((self.max_seqlen, 4), dtype=torch.float32)
        if seqlen > 1:
            decoder_target_seq_padded[:seqlen-1, :] = decoder_input_seq[1:seqlen, :]

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen_tensor = torch.tensor(seqlen, dtype=torch.int)
        mmsi =  torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0,4], dtype=torch.int)

        return encoder_input_density, decoder_input_seq_padded, decoder_target_seq_padded, mask, seqlen_tensor, mmsi, time_start

