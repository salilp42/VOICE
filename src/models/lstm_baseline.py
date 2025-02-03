import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from .base_model import BaseModel, FocalLoss
from .cnn_baseline import CNNBaseline

class Attention(nn.Module):
    """Attention mechanism for LSTM outputs"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Single query vector to compute attention
        self.query = nn.Parameter(torch.randn(hidden_size))
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output shape: [batch, seq_len, hidden_size]
        
        # Compute raw attention scores using dot product with query
        scores = torch.matmul(lstm_output, self.query)  # [batch, seq_len]
        
        # Scale scores for more distinct attention
        scores = scores * 5.0  # Increase scale factor for more pronounced differences
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        
        # Apply attention to get context
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # [batch, 1, hidden_size]
        context = context.squeeze(1)  # [batch, hidden_size]
        
        return context, attention_weights

class CustomLSTM(nn.LSTM):
    """Custom LSTM that can return internal states"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internal_states = []
    
    def forward(self, input, hx=None, return_states=False):
        self.internal_states = []
        
        # Get batch size and sequence length
        batch_size = input.size(0)
        seq_len = input.size(1)
        
        if hx is None:
            h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                           batch_size, self.hidden_size,
                           device=input.device)
            c0 = torch.zeros_like(h0)
            hx = (h0, c0)
        
        output, (hn, cn) = super().forward(input, hx)
        
        if return_states:
            return output, (hn, cn), self.internal_states
        return output, (hn, cn)

class LSTMBaseline(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        segment_length: int = 16000,
        base_filters: int = 32,
        hidden_size: int = 128,
        n_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
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
        
        self.conv4 = nn.Conv1d(base_filters*4, base_filters*8, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(base_filters*8)
        
        # Custom LSTM for temporal modeling
        self.lstm = CustomLSTM(
            input_size=base_filters*8,  # CNN output size
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = Attention(lstm_out_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, n_classes)
        )
        
        self.attention_weights = None  # Store attention weights for visualization
    
    def forward(self, batch: Dict[str, torch.Tensor], return_states: bool = False) -> Dict[str, Any]:
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
        
        # Prepare for LSTM: [batch_size, seq_len, features]
        features = x.transpose(1, 2)  # Change from [batch, channels, time] to [batch, time, channels]
        
        # Process through LSTM
        if return_states:
            lstm_out, (hn, cn), internal_states = self.lstm(features, return_states=True)
        else:
            lstm_out, (hn, cn) = self.lstm(features)
        
        # Apply attention
        context, attention_weights = self.attention(lstm_out)
        self.attention_weights = attention_weights  # Store for visualization
        
        # Classify using attended features
        logits = self.classifier(context)
        
        outputs = {
            'logits': logits,
            'features': context,
            'attention_weights': attention_weights
        }
        
        if return_states:
            outputs['internal_states'] = internal_states
            
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.forward(batch)
        return self.loss_fn(outputs['logits'], batch['label']) 