"""
MDDformer-Improved: Enhanced multimodal depression detection model.
===================================================================
Phase 2: Architectural improvements targeting >=80% accuracy.

Key improvements over the original MDDformer:
  1. Sinusoidal Positional Encoding for temporal sequences before cross-attention
  2. Stacked cross-attention layers (configurable depth, default=2)
  3. Pre-norm architecture (LayerNorm before attention, more stable training)
  4. GELU activation instead of ELU (better gradient flow for transformers)
  5. Increased FFN hidden dim (4x expansion instead of 2x)
  6. Configurable number of attention heads (default=8 instead of 4)
  7. Dropout in cross-attention for regularization
  8. Gradient-friendly residual connections throughout

All improvements are individually toggleable via the config for ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt


# ========================== Positional Encoding ==========================
class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ========================== TCN Backbone (unchanged) ==========================
class Conv1dBlock(nn.Module):
    """3-layer 1D CNN for initial video feature extraction."""
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(171, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class DilatedTCN(nn.Module):
    """Dilated Temporal Convolutional Network with residual connections."""
    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(dropout)

        self.conv1_1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv1_2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv2_1 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv4_1 = nn.Conv1d(128, 256, 3, padding=16, dilation=16)
        self.conv4_2 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)

    def forward(self, x):
        raw = x
        x = F.relu(self.conv1_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv1_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv2_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv2_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv3_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv4_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv4_2(x))
        raw = self.conv(raw)
        raw = F.relu(x + raw)
        return raw


class TCNModel(nn.Module):
    """TCN video feature encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.Conv1d = Conv1dBlock()
        self.DilatedTCN = DilatedTCN()

    def forward(self, input):
        input = input.transpose(1, 2)
        x = self.Conv1d(input)
        x = self.DilatedTCN(x)
        x = x.transpose(1, 2)
        return x


# ========================== Improved Cross-Attention ==========================
class ImprovedCalculateAttention(nn.Module):
    """
    Improved shared cross-attention with:
    - Attention dropout for regularization
    - Numerically stable softmax
    """
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, Qx, Kx, Vx, Qy, Ky, Vy):
        scale = sqrt(Qx.size(-1))
        
        attentionx = torch.matmul(Qx, torch.transpose(Kx, -1, -2))
        attentiony = torch.matmul(Qy, torch.transpose(Ky, -1, -2))
        
        attention = torch.cat((attentionx, attentiony), dim=1)
        B, C, H, W = attention.size()
        attention = attention.reshape(B, 2, C // 2, H, W)
        attention = torch.mean(attention, dim=1).squeeze()
        
        attention_weights = torch.softmax(attention / scale, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        attention1 = torch.matmul(attention_weights, Vx)
        attention2 = torch.matmul(attention_weights, Vy)
        return attention1, attention2


class ImprovedMultiCrossAttention(nn.Module):
    """
    Improved Multi-head bidirectional cross-attention with:
    - Configurable number of heads
    - Attention dropout
    - Output projection layer
    """
    def __init__(self, hidden_size, all_head_size, head_num, attn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        assert all_head_size % head_num == 0

        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        
        # Output projections (new)
        self.out_proj_x = nn.Linear(all_head_size, hidden_size)
        self.out_proj_y = nn.Linear(all_head_size, hidden_size)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        self.calc_attention = ImprovedCalculateAttention(attn_dropout)

    def forward(self, x, y):
        batch_size = x.size(0)
        
        q_sx = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sx = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sx = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        q_sy = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_sy = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_sy = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention1, attention2 = self.calc_attention(q_sx, k_sx, v_sx, q_sy, k_sy, v_sy)
        
        attention1 = attention1.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)
        attention2 = attention2.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)
        
        # Output projection + residual
        attention1 = self.proj_dropout(self.out_proj_x(attention1)) + x
        attention2 = self.proj_dropout(self.out_proj_y(attention2)) + y

        return attention1, attention2


# ========================== Improved Feed-Forward ==========================
class ImprovedFeedForward(nn.Module):
    """
    Improved FFN with GELU activation and larger expansion ratio.
    """
    def __init__(self, dim_in, hidden_dim, dim_out=None, dropout=0.1):
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=dim_in, out_channels=hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim, out_channels=dim_out, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ========================== Cross-Attention Block ==========================
class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block with pre-norm architecture:
      - LayerNorm -> Cross-Attention -> Residual
      - LayerNorm -> FFN -> Residual
    """
    def __init__(self, hidden_size=128, head_num=8, ffn_expansion=4,
                 attn_dropout=0.1, ffn_dropout=0.1, proj_dropout=0.1):
        super().__init__()
        self.norm1_x = nn.LayerNorm(hidden_size)
        self.norm1_y = nn.LayerNorm(hidden_size)
        
        self.cross_attn = ImprovedMultiCrossAttention(
            hidden_size=hidden_size,
            all_head_size=hidden_size,
            head_num=head_num,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout
        )
        
        self.norm2_x = nn.LayerNorm(hidden_size)
        self.norm2_y = nn.LayerNorm(hidden_size)
        
        self.ffn_x = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_expansion),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * ffn_expansion, hidden_size),
            nn.Dropout(ffn_dropout),
        )
        self.ffn_y = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_expansion),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden_size * ffn_expansion, hidden_size),
            nn.Dropout(ffn_dropout),
        )
    
    def forward(self, x, y):
        # Cross-attention with pre-norm
        x_normed = self.norm1_x(x)
        y_normed = self.norm1_y(y)
        x, y = self.cross_attn(x_normed, y_normed)
        
        # FFN with pre-norm
        x = x + self.ffn_x(self.norm2_x(x))
        y = y + self.ffn_y(self.norm2_y(y))
        
        return x, y


# ========================== Video Projection ==========================
class ConvNet1d(nn.Module):
    """Linear projection from TCN output (256-dim) to 128-dim."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(256, 128)

    def forward(self, input):
        sizeTmp = input.size(1)
        batch_size = input.size(0)
        outConv1d = input.contiguous().view(input.size(0) * input.size(1), -1)
        output = self.fc(outConv1d)
        output = output.view(batch_size, sizeTmp, -1)
        return output


# ========================== Improved Classifier ==========================
class ImprovedClassifier(nn.Module):
    """
    Enhanced classification head with:
    - LayerNorm before classification
    - GELU activation
    - Configurable hidden dim and dropout
    """
    def __init__(self, input_dim=186, hidden_dim=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ========================== MDDformer-Improved ==========================
class MDDformerImproved(nn.Module):
    """
    MDDformer-Improved: Enhanced multimodal depression detection model.
    
    Improvements:
      1. Positional encoding before cross-attention
      2. Stacked cross-attention layers (configurable depth)
      3. Pre-norm architecture
      4. GELU activation
      5. Attention & projection dropout
      6. Configurable heads, FFN expansion
      7. Enhanced classifier head
    
    Args:
        num_cross_attn_layers: Number of stacked cross-attention blocks (default: 2)
        head_num: Number of attention heads (default: 8)
        ffn_expansion: FFN hidden dim multiplier (default: 4)
        attn_dropout: Dropout in attention weights (default: 0.1)
        ffn_dropout: Dropout in FFN layers (default: 0.1)
        proj_dropout: Dropout after output projection (default: 0.1)
        cls_dropout: Dropout in classifier (default: 0.2)
        use_pos_encoding: Whether to use positional encoding (default: True)
        pos_encoding_type: 'sinusoidal' or 'learnable' (default: 'sinusoidal')
    """
    def __init__(self, 
                 num_cross_attn_layers=2,
                 head_num=8,
                 ffn_expansion=4,
                 attn_dropout=0.1,
                 ffn_dropout=0.1,
                 proj_dropout=0.1,
                 cls_dropout=0.2,
                 use_pos_encoding=True,
                 pos_encoding_type='sinusoidal'):
        super().__init__()
        
        hidden_size = 128  # Matching audio feature dim
        
        # Video branch
        self.TCNModel = TCNModel()
        self.Conv1dModel = ConvNet1d()
        self.conv = nn.Conv1d(in_channels=114, out_channels=186, kernel_size=1, padding=0, stride=1)
        
        # Positional encoding
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            if pos_encoding_type == 'sinusoidal':
                self.pos_enc_video = SinusoidalPositionalEncoding(hidden_size, max_len=200, dropout=0.1)
                self.pos_enc_audio = SinusoidalPositionalEncoding(hidden_size, max_len=200, dropout=0.1)
            else:
                self.pos_enc_video = LearnablePositionalEncoding(hidden_size, max_len=200, dropout=0.1)
                self.pos_enc_audio = LearnablePositionalEncoding(hidden_size, max_len=200, dropout=0.1)
        
        # Stacked cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                hidden_size=hidden_size,
                head_num=head_num,
                ffn_expansion=ffn_expansion,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                proj_dropout=proj_dropout
            )
            for _ in range(num_cross_attn_layers)
        ])
        
        # Final fusion
        self.final_norm = nn.LayerNorm(hidden_size * 2)
        self.fusion_ffn = ImprovedFeedForward(
            dim_in=186, hidden_dim=186 * ffn_expansion, dim_out=186, dropout=ffn_dropout
        )
        self.final_norm2 = nn.LayerNorm(hidden_size * 2)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = ImprovedClassifier(
            input_dim=186, hidden_dim=128, num_classes=2, dropout=cls_dropout
        )
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, inputVideo, inputAudio):
        """
        Args:
            inputVideo: (B, 915, 171) - TCN-processed video features
            inputAudio: (B, 186, 128) - VGGish audio features
        Returns:
            (B, 2) - classification probabilities
        """
        # Video branch: TCN -> Linear projection -> temporal alignment
        video = self.TCNModel(inputVideo)           # (B, 114, 256)
        video = self.Conv1dModel(video)              # (B, 114, 128)
        video = self.conv(video)                     # (B, 186, 128)
        
        audio = inputAudio                           # (B, 186, 128)
        
        # Add positional encoding
        if self.use_pos_encoding:
            video = self.pos_enc_video(video)
            audio = self.pos_enc_audio(audio)
        
        # Stacked cross-attention layers
        for cross_attn_block in self.cross_attn_layers:
            video, audio = cross_attn_block(video, audio)
        
        # Concatenate attended features
        output = torch.cat((video, audio), dim=2)    # (B, 186, 256)
        
        # Final FFN with residual
        output = self.fusion_ffn(self.final_norm(output)) + output
        output = self.final_norm2(output)
        
        # Pool and classify
        output = self.pooling(output).reshape(output.shape[0], -1)  # (B, 186)
        result = self.classifier(output)
        result = self.softmax(result)
        
        return result


if __name__ == '__main__':
    # Test the improved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MDDformerImproved(
        num_cross_attn_layers=2,
        head_num=8,
        ffn_expansion=4,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        proj_dropout=0.1,
        cls_dropout=0.2,
        use_pos_encoding=True,
        pos_encoding_type='sinusoidal'
    ).to(device)
    
    x1 = torch.randn(4, 186, 128).to(device)   # Audio
    x2 = torch.randn(4, 915, 171).to(device)    # Video
    y = model(x2, x1)
    print(f"Output shape: {y.shape}")   # Expected: (4, 2)
    print(f"Output sample: {y[0]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compare with original
    from MDDformermodel import Net
    original = Net().to(device)
    orig_params = sum(p.numel() for p in original.parameters())
    print(f"\nOriginal MDDformer params: {orig_params:,}")
    print(f"Improved MDDformer params: {total_params:,}")
    print(f"Parameter increase: {(total_params - orig_params) / orig_params * 100:.1f}%")
