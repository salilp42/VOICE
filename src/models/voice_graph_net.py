import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
from torch_geometric.nn import GATConv
import torch_geometric.nn as geom_nn
import numpy as np

from .base_model import BaseModel

class CustomGATConv(GATConv):
    """Custom GAT layer that can return attention weights"""
    def forward(self, x, edge_index, return_attention=False):
        if isinstance(x, Tuple):
            x = (None if x[0] is None else x[0].unsqueeze(-1) if x[0].dim() == 1 else x[0],
                 None if x[1] is None else x[1].unsqueeze(-1) if x[1].dim() == 1 else x[1])
        else:
            x = None if x is None else x.unsqueeze(-1) if x.dim() == 1 else x
        
        # Run through parent's forward
        out = super().forward(x, edge_index)
        
        if return_attention:
            # Get attention weights from the last self-attention layer
            alpha = self._alpha if hasattr(self, '_alpha') else torch.ones(edge_index.size(1), device=edge_index.device)
            return out, alpha
        return out

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class VoiceGraphNet(BaseModel):
    """Lightweight Graph Neural Network for voice-based PD detection with domain adaptation."""
    def __init__(self, hidden_dim=128, temperature=0.2, topk=6, sim_threshold=0.3):
        super().__init__()
        
        # Hyperparameters
        self.temperature = temperature
        self.topk = topk
        self.sim_threshold = sim_threshold
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_dim//4, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim//2, hidden_dim//2, 1),
            nn.ReLU()
        )
        
        # Domain adaptation
        self.domain_adapter = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Graph layers with custom GAT
        gat_hidden = hidden_dim//4  # Reduced hidden dimension for GAT layers
        self.gat1 = CustomGATConv(hidden_dim//2, gat_hidden, heads=2, dropout=0.2)
        self.gat2 = CustomGATConv(gat_hidden*2, hidden_dim//2, heads=1, dropout=0.2)
        
        # Normalization layers
        self.input_norm = nn.InstanceNorm1d(1)
        self.gat1_norm = nn.InstanceNorm1d(gat_hidden*2)
        self.gat2_norm = nn.InstanceNorm1d(hidden_dim//2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 2)
        )
        
        # Store attention weights and edge information
        self.temporal_attention = None
        self.similarity_attention = None
        self.edge_index = None
        self.edge_type = None  # 0 for temporal, 1 for similarity
    
    def _build_graph(self, features: torch.Tensor, participant_ids: list) -> tuple:
        device = features.device
        edge_index = []
        edge_type = []  # Track edge types
        temporal_weights = []  # Store temporal edge weights
        similarity_weights = []  # Store similarity edge weights
        
        # Dynamic threshold based on feature statistics
        feature_mean = features.mean(dim=1, keepdim=True)
        feature_std = features.std(dim=1, keepdim=True)
        normalized_features = (features - feature_mean) / (feature_std + 1e-5)
        
        # Ensure participant_ids is a flat list
        if isinstance(participant_ids[0], list):
            participant_ids = [pid for sublist in participant_ids for pid in sublist]
        
        # Get unique participant IDs
        unique_pids = list(set(str(pid) for pid in participant_ids))
        
        for pid in unique_pids:
            indices = [i for i, p in enumerate(participant_ids) if str(p) == pid]
            indices.sort()
            
            # Sequential edges
            for i in range(len(indices) - 1):
                edge_index.append([indices[i], indices[i + 1]])
                edge_index.append([indices[i + 1], indices[i]])
                edge_type.extend([0, 0])  # Temporal edges
                temporal_weights.extend([1.0, 1.0])  # Default weight
            
            # Similarity-based edges
            if len(indices) > 2:
                participant_features = normalized_features[indices]
                
                cos_sim = F.cosine_similarity(participant_features.unsqueeze(1), 
                                           participant_features.unsqueeze(0), dim=2)
                l2_dist = torch.cdist(participant_features, participant_features)
                l2_sim = 1 / (1 + l2_dist)
                
                sim = (cos_sim + l2_sim) / 2
                sim = sim / self.temperature
                
                sim_mean = sim.mean()
                sim_std = sim.std()
                adaptive_threshold = sim_mean + 0.5 * sim_std
                
                _, topk_indices = sim.topk(k=min(self.topk, len(indices)), dim=1)
                
                for i, similar_indices in enumerate(topk_indices):
                    for j in similar_indices[1:]:
                        if abs(i - j.item()) > 1:
                            sim_value = sim[i, j].item()
                            if sim_value > max(self.sim_threshold, adaptive_threshold):
                                edge_index.append([indices[i], indices[j.item()]])
                                edge_index.append([indices[j.item()], indices[i]])
                                edge_type.extend([1, 1])  # Similarity edges
                                similarity_weights.extend([sim_value, sim_value])
        
        if not edge_index:  # If no edges were created
            # Create a simple chain of temporal edges
            for i in range(features.size(0) - 1):
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
                edge_type.extend([0, 0])
                temporal_weights.extend([1.0, 1.0])
        
        edge_index = torch.tensor(edge_index, device=device).t().contiguous()
        edge_type = torch.tensor(edge_type, device=device)
        temporal_weights = torch.tensor(temporal_weights, device=device)
        similarity_weights = torch.tensor(similarity_weights, device=device)
        
        # Store for visualization
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.temporal_attention = temporal_weights
        self.similarity_attention = similarity_weights
        
        return edge_index
    
    def forward(self, batch: Dict[str, torch.Tensor], return_attention: bool = False) -> Dict[str, Any]:
        # Extract features
        x = self.input_norm(batch['segment'].unsqueeze(1))
        x = self.feature_extractor(x).squeeze(-1)  # [B, hidden_dim//2]
        x_initial = x
        
        # Domain adaptation with gradient reversal
        x = grad_reverse(x, lambda_=0.3)
        x = self.domain_adapter(x)  # [B, hidden_dim//2]
        
        # Graph processing
        edge_index = self._build_graph(x, batch['participant_id'])
        
        # First GAT layer with attention weights
        if return_attention:
            x, attn1 = self.gat1(x, edge_index, return_attention=True)
        else:
            x = self.gat1(x, edge_index)
        x = self.gat1_norm(x.unsqueeze(1)).squeeze(1)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GAT layer with attention weights
        if return_attention:
            x, attn2 = self.gat2(x, edge_index, return_attention=True)
        else:
            x = self.gat2(x, edge_index)
        x = self.gat2_norm(x.unsqueeze(1)).squeeze(1)
        x = F.relu(x)
        
        # Residual connection
        x = x + x_initial
        
        # Classification
        logits = self.classifier(x)
        
        outputs = {
            'logits': logits,
            'features': x
        }
        
        if return_attention:
            attention_info = {
                'edge_index': self.edge_index,
                'edge_type': self.edge_type,
                'temporal_attention': self.temporal_attention,
                'similarity_attention': self.similarity_attention,
                'gat1_attention': attn1,
                'gat2_attention': attn2
            }
            outputs['attention_info'] = attention_info
            
        return outputs
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.forward(batch)
        return self.loss_fn(outputs['logits'], batch['label'])
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            loss = self.compute_loss(batch)
            
            segment_probs = F.softmax(outputs['logits'], dim=1)[:, 1]
            
            if not hasattr(self, 'all_predictions') or isinstance(self.all_predictions, torch.Tensor):
                self.all_predictions = []
                self.all_labels = []
                self.all_probabilities = []
            
            for i, (participant_id, prob, label) in enumerate(zip(batch['participant_id'], 
                                                                segment_probs, 
                                                                batch['label'])):
                self.participant_predictions[participant_id].append(prob.item())
                self.participant_labels[participant_id] = label.item()
                
                self.all_predictions.append((prob > 0.5).long().item())
                self.all_labels.append(label.item())
                self.all_probabilities.append(prob.item())
            
            return {'val_loss': loss}
    
    def on_validation_end(self):
        if not isinstance(self.all_predictions, torch.Tensor):
            self.all_predictions = torch.tensor(self.all_predictions)
            self.all_labels = torch.tensor(self.all_labels)
            self.all_probabilities = torch.tensor(self.all_probabilities)

    def get_initial_features(self, batch):
        """Get initial node features before GNN processing."""
        x = self.input_norm(batch['segment'].unsqueeze(1))
        x = self.feature_extractor(x)
        x = x.reshape(x.size(0), -1)  # Flatten features
        return x

    def get_all_intermediate_features(self, batch):
        """Get features from each stage of the model."""
        # Initial CNN features
        x = self.input_norm(batch['segment'].unsqueeze(1))
        x = self.feature_extractor(x)
        initial_features = x.reshape(x.size(0), -1)
        
        # Domain adaptation features
        x = grad_reverse(initial_features, lambda_=0.3)
        domain_features = self.domain_adapter(x)
        
        # First GAT layer features
        edge_index = self._build_graph(domain_features, batch['participant_id'])
        x, attn1 = self.gat1(domain_features, edge_index, return_attention=True)
        x = self.gat1_norm(x.unsqueeze(1)).squeeze(1)
        x = F.relu(x)
        gat1_features = F.dropout(x, p=0.2, training=self.training)
        
        # Second GAT layer features
        x, attn2 = self.gat2(gat1_features, edge_index, return_attention=True)
        x = self.gat2_norm(x.unsqueeze(1)).squeeze(1)
        x = F.relu(x)
        final_features = x + initial_features  # Include residual connection
        
        return {
            'initial': initial_features,
            'domain': domain_features,
            'gat1': gat1_features,
            'final': final_features,
            'edge_index': edge_index,
            'edge_type': self.edge_type,
            'attention': {
                'gat1': attn1,
                'gat2': attn2
            }
        }