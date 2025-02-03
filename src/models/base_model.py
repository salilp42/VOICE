import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from collections import defaultdict

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.participant_predictions = defaultdict(list)
        self.participant_labels = {}
        self.loss_fn = FocalLoss()
        self.min_segments_threshold = 5  # Minimum segments needed for reliable prediction
        
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss for a batch of data."""
        raise NotImplementedError
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        raise NotImplementedError
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a training step."""
        self.train()
        loss = self.compute_loss(batch)
        return {'loss': loss}
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a validation step and accumulate participant-level predictions."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            loss = self.compute_loss(batch)
            
            # Store predictions and labels by participant
            probs = F.softmax(outputs['logits'], dim=1)[:, 1]  # Probability of PD
            for i, participant_id in enumerate(batch['participant_id']):
                self.participant_predictions[participant_id].append(probs[i].item())
                self.participant_labels[participant_id] = batch['label'][i].item()
            
            return {'val_loss': loss}
    
    def compute_participant_metrics(self) -> Dict[str, Any]:
        """Compute participant-level metrics with improved aggregation."""
        participant_metrics = []
        valid_participants = []
        
        # Compute metrics for each participant
        for participant_id in self.participant_predictions:
            predictions = self.participant_predictions[participant_id]
            true_label = self.participant_labels[participant_id]
            
            if len(predictions) < self.min_segments_threshold:
                continue  # Skip participants with too few segments
                
            # Compute prediction confidence (distance from decision boundary)
            confidences = [abs(p - 0.5) for p in predictions]
            confidence_weights = np.array(confidences) / sum(confidences)
            
            # Weighted average probability (weighted by confidence)
            avg_prob = np.sum(np.array(predictions) * confidence_weights)
            pred_label = int(avg_prob > 0.5)
            
            metrics = {
                'participant_id': participant_id,
                'true_label': true_label,
                'predicted_label': pred_label,
                'average_probability': avg_prob,
                'accuracy': float(pred_label == true_label),
                'segment_predictions': predictions,
                'num_segments': len(predictions),
                'prediction_confidence': np.mean(confidences)
            }
            participant_metrics.append(metrics)
            valid_participants.append(metrics)
        
        # Aggregate metrics only for valid participants
        n_participants = len(valid_participants)
        if n_participants == 0:
            return {
                'participant_metrics': [],
                'participant_accuracy': 0.0,
                'participant_auc': 0.5,
                'valid_participants': 0
            }
        
        # Calculate overall participant-level accuracy
        participant_accuracy = np.mean([m['accuracy'] for m in valid_participants])
        
        # Calculate participant-level AUC using confidence-weighted probabilities
        try:
            participant_auc = roc_auc_score(
                [m['true_label'] for m in valid_participants],
                [m['average_probability'] for m in valid_participants]
            )
        except:
            participant_auc = 0.5
            
        return {
            'participant_metrics': participant_metrics,
            'participant_accuracy': float(participant_accuracy),
            'participant_auc': float(participant_auc),
            'valid_participants': n_participants
        }

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smooth=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smooth = label_smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply label smoothing
        if self.label_smooth > 0:
            n_classes = inputs.size(-1)
            targets_one_hot = F.one_hot(targets, n_classes).float()
            targets_smooth = (1 - self.label_smooth) * targets_one_hot + self.label_smooth / n_classes
            ce_loss = -(targets_smooth * F.log_softmax(inputs, dim=-1)).sum(dim=-1)
            pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() 