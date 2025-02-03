import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .base_model import BaseModel, FocalLoss

class CNNBaseline(BaseModel):
    def __init__(
        self,
        in_channels: int = 1,
        segment_length: int = 16000,
        base_filters: int = 32,
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.loss_fn = FocalLoss()
        
        # Individual CNN layers for easier access
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=128, stride=4, padding=64)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(base_filters, base_filters*2, kernel_size=64, stride=2, padding=32)
        self.bn2 = nn.BatchNorm1d(base_filters*2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(base_filters*2, base_filters*4, kernel_size=32, stride=2, padding=16)
        self.bn3 = nn.BatchNorm1d(base_filters*4)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop3 = nn.Dropout(dropout)
        
        self.conv4 = nn.Conv1d(base_filters*4, base_filters*8, kernel_size=16, stride=1, padding=8)
        self.bn4 = nn.BatchNorm1d(base_filters*8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_filters*8, n_classes)
        )
        
        # Store intermediate activations
        self.intermediate_activations = {}
    
    def get_conv_layer(self, layer_idx: int) -> nn.Conv1d:
        """Get a specific convolutional layer for Grad-CAM."""
        if layer_idx == 1: return self.conv1
        elif layer_idx == 2: return self.conv2
        elif layer_idx == 3: return self.conv3
        elif layer_idx == 4: return self.conv4
        else: raise ValueError(f"Invalid layer index: {layer_idx}")
    
    def forward(self, batch: Dict[str, torch.Tensor], return_activations: bool = False) -> Dict[str, Any]:
        # Get waveform segments [batch_size, segment_length]
        x = batch['segment']
        
        # Add channel dimension [batch_size, 1, segment_length]
        x = x.unsqueeze(1)
        
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if return_activations:
            self.intermediate_activations['conv1'] = x.detach()
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        if return_activations:
            self.intermediate_activations['conv2'] = x.detach()
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        if return_activations:
            self.intermediate_activations['conv3'] = x.detach()
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        if return_activations:
            self.intermediate_activations['conv4'] = x.detach()
        
        # Global pooling
        features = self.global_pool(x)
        features = features.squeeze(-1)
        
        # Classify
        logits = self.classifier(features)
        
        outputs = {
            'logits': logits,
            'features': features
        }
        
        if return_activations:
            outputs['activations'] = self.intermediate_activations
            
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.forward(batch)
        return self.loss_fn(outputs['logits'], batch['label']) 