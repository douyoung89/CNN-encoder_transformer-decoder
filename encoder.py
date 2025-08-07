import math
import torch
import torch.nn as nn
from torch.nn import functional as F

def init_weights(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

class Encoder(nn.Module):
    """
    어떤 H×W 입력이 와도 (batch, seq_len, n_embd) 로 나오는 CNN+Pooling 인코더
    """
    def __init__(self, config):
        super().__init__()
        mid_ch = config.n_embd // 2
        in_ch  = 3  # now density + lat + lon

        # 1) 두 스텝의 Conv→GELU
        #    (원하면 더 쌓아도 되고, kernel_size=3을 써도 됩니다)
        self.cnn = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=(3,3,3), padding=(1,1,1)),  # H,W 유지 → mid_ch
            nn.BatchNorm3d(mid_ch),
            nn.GELU(),
            
            nn.Conv3d(mid_ch, mid_ch, kernel_size=(3,3,3), padding=(1,1,1)),  # H,W 유지 → n_embd
            nn.BatchNorm3d(mid_ch),
            nn.GELU(),
            
            nn.Conv3d(mid_ch, config.n_embd, kernel_size=(3,3,3), padding=(1,1,1)),  # H,W 유지 → n_embd
            nn.BatchNorm3d(config.n_embd),
            nn.GELU(),
            
            # nn.Conv3d(mid_ch, config.n_embd, kernel_size=(3,3,3), padding=(1,1,1)),  # H,W 유지 → n_embd
            # nn.BatchNorm3d(config.n_embd),
            # nn.GELU(),
            # nn.Conv2d(mid_ch, config.n_embd, kernel_size=3, padding=1),  # H,W 유지 → n_embd
            # nn.GELU(),
        )
        
        # 2) Adaptive 풀링으로 H×W → 1×1
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
        # transformer 
        layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4*config.n_embd,
            dropout=config.resid_pdrop,
            activation='gelu',
            batch_first=True  # so input is (B, T, C)
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layer)

        # 3) Positional Embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen+1, config.n_embd))
        self.drop    = nn.Dropout(config.embd_pdrop)

        self.apply(init_weights)
        print(f"[CNNEncoder] params: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            density: (batch, seq_len, C, H, W)
        Returns:
            (batch, seq_len, n_embd)
        """
        # 3 ch -------------
        b, t, C, H, W = x.size()
        # → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # 3D CNN + spatial pooling
        # x: (B, n_embd, T, 1, 1)
        x = self.cnn(x)
        x = self.pool(x)

        # → (B, T, n_embd)
        x = x.squeeze(-1).squeeze(-1)       # (B, n_embd, T)
        x = x.permute(0, 2, 1).contiguous() # (B, T, n_embd)

        # add positional, drop, then transformer
        pos = self.pos_emb[:, :t, :]        # (1, T, n_embd)
        x = self.drop(x + pos)
        
        
        # 1ch ----------------------
        # b, t, H, W = x.size()
        # x = x.unsqueeze(1) # b, 1, t, h, w
        # x = self.cnn(x)
        # x = self.pool(x) # (B, n_embd, T, 1, 1)
        # x = x.squeeze(-1).squeeze(-1)       # (B, n_embd, T)
        # x = x.permute(0, 2, 1).contiguous() # (B, T, n_embd)
        # pos = self.pos_emb[:, :t, :]        # (1, T, n_embd)
        # x = self.drop(x + pos)
        return x       # (B, T, n_embd)