import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .base_model import BaseModel, FocalLoss
from .cnn_baseline import CNNBaseline

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Store for visualization
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        self.attention_weights = attention.detach()  # Store for visualization
        attention = self.dropout(attention)
        
        # Apply attention to values
        x = torch.matmul(attention, V)
        
        # Reshape and project to output
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.W_o(x)
        
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attended = self.attention(x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        
        return x

class TransformerBaseline(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        segment_length: int = 16000,
        base_filters: int = 32,
        n_layers: int = 4,
        n_heads: int = 8,
        d_model: int = 256,
        d_ff: int = 1024,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.loss_fn = FocalLoss()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(base_filters*2)
        self.pool2 = nn.MaxPool1d(4)
        self.drop2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(base_filters*4)
        self.pool3 = nn.MaxPool1d(4)
        self.drop3 = nn.Dropout(dropout)
        
        self.conv4 = nn.Conv1d(base_filters*4, d_model, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(d_model)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        # Get waveform segments [batch_size, segment_length]
        x = batch['segment']
        
        # Add channel dimension [batch_size, 1, segment_length]
        x = x.unsqueeze(1)
        
        # Extract CNN features
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Prepare for Transformer: [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)  # Change from [batch, channels, time] to [batch, time, channels]
        
        # Store attention weights for visualization
        self.attention_weights = []
        
        # Process through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            self.attention_weights.append(block.attention.attention_weights)
        
        # Use first token (CLS) for classification
        x = x[:, 0]
        
        # Classify
        logits = self.classifier(x)
        
        outputs = {
            'logits': logits,
            'features': x,
            'attention_weights': self.attention_weights
        }
        
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.forward(batch)
        return self.loss_fn(outputs['logits'], batch['label']) 