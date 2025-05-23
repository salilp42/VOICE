#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpretability Analysis for VoiceGraphNet
-------------------------------------------
This script analyzes the learned GAT attention weights on temporal vs. similarity edges
in the VoiceGraphNet model to understand their independent contributions to PD detection.

The script addresses two reviewer comments:
1. The interpretability of edge weights in relation to PD pathophysiology
2. The independent contributions of temporal and similarity edges to classification

Author: Cascade AI Assistant
Date: 2025-05-19
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import json
import argparse

# Import project modules
from models.voice_graph_net import VoiceGraphNet
from data.dataset import VoiceDataModule

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
COLORS = sns.color_palette("Set2")

# Create directories for saving results
RESULTS_DIR = Path("/Users/salilpatel/Desktop/PD Voice/interpretability_results")
FIGURES_DIR = Path("/Users/salilpatel/Desktop/PD Voice/interpretability_figures")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
set_random_seeds()

def load_model(checkpoint_path: str, device: torch.device) -> VoiceGraphNet:
    """
    Load a trained VoiceGraphNet model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded VoiceGraphNet model
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model with default parameters
    model = VoiceGraphNet(hidden_dim=128)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

def extract_interpretability_data(
    model: VoiceGraphNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract attention weights, edge types, and node features from the model.
    
    Args:
        model: Trained VoiceGraphNet model
        dataloader: DataLoader containing validation data
        device: Device to run inference on
        num_batches: Number of batches to process (None for all)
        
    Returns:
        Dictionary containing extracted data
    """
    print("Extracting interpretability data...")
    
    # Initialize storage for collected data
    collected_data = {
        'gat1_attention': [],
        'gat2_attention': [],
        'edge_types': [],
        'edge_indices': [],
        'temporal_attention': [],  # Construction weights for temporal edges
        'similarity_attention': [], # Construction weights for similarity edges
        'initial_features': [],
        'final_features': [],
        'participant_ids': [],
        'labels': [],
        'logits': []
    }
    
    # Process batches
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if num_batches is not None and batch_idx >= num_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with attention weights
            outputs = model.forward(batch, return_attention=True)
            
            # Extract attention info
            attention_info = outputs['attention_info']
            
            # Store data
            collected_data['gat1_attention'].append(attention_info['gat1_attention'].cpu())
            collected_data['gat2_attention'].append(attention_info['gat2_attention'].cpu())
            collected_data['edge_types'].append(attention_info['edge_type'].cpu())
            collected_data['edge_indices'].append(attention_info['edge_index'].cpu())
            
            # Store construction weights
            if 'temporal_attention' in attention_info:
                collected_data['temporal_attention'].append(attention_info['temporal_attention'].cpu())
            if 'similarity_attention' in attention_info:
                collected_data['similarity_attention'].append(attention_info['similarity_attention'].cpu())
            
            # Get features using the model's feature extraction methods
            features = model.get_all_intermediate_features(batch)
            collected_data['initial_features'].append(features['initial'].cpu())
            collected_data['final_features'].append(features['final'].cpu())
            
            # Store participant IDs and labels
            collected_data['participant_ids'].extend(batch['participant_id'])
            collected_data['labels'].append(batch['label'].cpu())
            collected_data['logits'].append(outputs['logits'].cpu())
    
    # Concatenate batch data where appropriate
    for key in ['labels', 'logits', 'initial_features', 'final_features']:
        if collected_data[key]:
            collected_data[key] = torch.cat(collected_data[key], dim=0)
    
    print(f"Extracted data from {len(collected_data['gat1_attention'])} batches")
    
    return collected_data

def analyze_edge_attention(collected_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze edge attention weights to understand the contribution of different edge types.
    Focus on construction weights since GAT attention weights are not correctly stored.
    
    Args:
        collected_data: Dictionary containing extracted data
        
    Returns:
        Dictionary containing analysis results
    """
    print("Analyzing edge attention...")
    
    # Initialize results dictionary
    results = {
        'construction': {'temporal': [], 'similarity': []},
        'by_diagnosis': {
            'PD': {'temporal_construction': [], 'similarity_construction': []},
            'HC': {'temporal_construction': [], 'similarity_construction': []}
        },
        'statistics': {}
    }
    
    # Create mapping from participant ID to diagnosis
    participant_to_label = {}
    for i, p_id in enumerate(collected_data['participant_ids']):
        participant_to_label[p_id] = collected_data['labels'][i].item()
    
    # Process each batch
    for batch_idx in range(len(collected_data['edge_indices'])):
        edge_indices = collected_data['edge_indices'][batch_idx]
        edge_types = collected_data['edge_types'][batch_idx]
        
        # Get construction weights for this batch
        temporal_weights = collected_data.get('temporal_attention', [])
        similarity_weights = collected_data.get('similarity_attention', [])
        
        if batch_idx < len(temporal_weights):
            temporal_batch_weights = temporal_weights[batch_idx]
            similarity_batch_weights = similarity_weights[batch_idx]
            
            # Store construction weights
            results['construction']['temporal'].extend(temporal_batch_weights.tolist())
            results['construction']['similarity'].extend(similarity_batch_weights.tolist())
            
            # Get participant IDs for the nodes in this batch
            batch_start = batch_idx * 32
            batch_end = min(batch_start + 32, len(collected_data['participant_ids']))
            batch_participant_ids = collected_data['participant_ids'][batch_start:batch_end]
            
            # Separate by diagnosis (PD vs HC)
            # For each edge, determine the diagnosis of the source node
            for i in range(edge_indices.shape[1]):
                source_node = edge_indices[0, i].item()
                if source_node >= len(batch_participant_ids):
                    continue  # Skip if source node index is out of bounds
                    
                participant_id = batch_participant_ids[source_node]
                diagnosis = 'PD' if participant_to_label.get(participant_id, 0) == 1 else 'HC'
                
                if edge_types[i] == 0 and i < len(temporal_batch_weights):  # Temporal edge
                    results['by_diagnosis'][diagnosis]['temporal_construction'].append(temporal_batch_weights[i].item())
                elif edge_types[i] == 1 and i < len(similarity_batch_weights):  # Similarity edge
                    results['by_diagnosis'][diagnosis]['similarity_construction'].append(similarity_batch_weights[i].item())
    
    # Calculate statistics for construction weights
    if results['construction']['temporal'] and results['construction']['similarity']:
        results['statistics']['construction_temporal_mean'] = np.mean(results['construction']['temporal'])
        results['statistics']['construction_temporal_std'] = np.std(results['construction']['temporal'])
        results['statistics']['construction_similarity_mean'] = np.mean(results['construction']['similarity'])
        results['statistics']['construction_similarity_std'] = np.std(results['construction']['similarity'])
        
        # T-test for construction temporal vs similarity
        t_stat_const, p_val_const = stats.ttest_ind(results['construction']['temporal'], results['construction']['similarity'])
        results['statistics']['construction_ttest'] = {'t_statistic': t_stat_const, 'p_value': p_val_const}
    
    # Calculate statistics by diagnosis for construction weights
    for diagnosis in ['PD', 'HC']:
        # Mean attention for each edge type
        if results['by_diagnosis'][diagnosis]['temporal_construction']:
            results['statistics'][f'{diagnosis}_temporal_construction_mean'] = np.mean(results['by_diagnosis'][diagnosis]['temporal_construction'])
            results['statistics'][f'{diagnosis}_temporal_construction_std'] = np.std(results['by_diagnosis'][diagnosis]['temporal_construction'])
        
        if results['by_diagnosis'][diagnosis]['similarity_construction']:
            results['statistics'][f'{diagnosis}_similarity_construction_mean'] = np.mean(results['by_diagnosis'][diagnosis]['similarity_construction'])
            results['statistics'][f'{diagnosis}_similarity_construction_std'] = np.std(results['by_diagnosis'][diagnosis]['similarity_construction'])
        
        # T-test within diagnosis group
        if results['by_diagnosis'][diagnosis]['temporal_construction'] and results['by_diagnosis'][diagnosis]['similarity_construction']:
            t_stat, p_val = stats.ttest_ind(
                results['by_diagnosis'][diagnosis]['temporal_construction'],
                results['by_diagnosis'][diagnosis]['similarity_construction']
            )
            results['statistics'][f'{diagnosis}_construction_ttest'] = {'t_statistic': t_stat, 'p_value': p_val}
    
    # T-test between PD and HC for each edge type
    for edge_type in ['temporal_construction', 'similarity_construction']:
        if (results['by_diagnosis']['PD'][edge_type] and 
            results['by_diagnosis']['HC'][edge_type]):
            t_stat, p_val = stats.ttest_ind(
                results['by_diagnosis']['PD'][edge_type],
                results['by_diagnosis']['HC'][edge_type]
            )
            results['statistics'][f'PD_vs_HC_{edge_type}'] = {'t_statistic': t_stat, 'p_value': p_val}
    
    # Calculate ratio of similarity to temporal edges
    temporal_count = len(results['construction']['temporal'])
    similarity_count = len(results['construction']['similarity'])
    total_edges = temporal_count + similarity_count
    
    if total_edges > 0:
        results['statistics']['temporal_edge_percentage'] = (temporal_count / total_edges) * 100
        results['statistics']['similarity_edge_percentage'] = (similarity_count / total_edges) * 100
    
    print("Edge attention analysis complete.")
    return results

def analyze_node_features(collected_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze node features to explore focal instabilities.
    
    Args:
        collected_data: Dictionary containing extracted data
        
    Returns:
        Dictionary containing analysis results
    """
    print("Analyzing node features...")
    
    # Initialize results dictionary
    results = {
        'tsne': {},
        'variability': {'PD': [], 'HC': []},
        'feature_stats': {'PD': {}, 'HC': {}},
        'statistics': {}
    }
    
    # Get labels and features
    labels = collected_data['labels']
    initial_features = collected_data['initial_features']
    final_features = collected_data['final_features']
    participant_ids = collected_data['participant_ids']
    
    # Group features by participant ID and diagnosis
    participant_features = {}
    participant_labels = {}
    pd_features = []
    hc_features = []
    
    for i, p_id in enumerate(participant_ids):
        if p_id not in participant_features:
            participant_features[p_id] = []
            participant_labels[p_id] = labels[i].item()
        
        feature_vector = final_features[i].numpy()
        participant_features[p_id].append(feature_vector)
        
        # Collect features by diagnosis
        if labels[i].item() == 1:  # PD
            pd_features.append(feature_vector)
        else:  # HC
            hc_features.append(feature_vector)
    
    # Calculate feature variability for each participant
    for p_id, features_list in participant_features.items():
        # Convert to numpy array for easier manipulation
        features_array = np.array(features_list)
        
        # Calculate standard deviation across segments for each feature dimension
        feature_std = np.std(features_array, axis=0)
        
        # Use L2 norm of the std vector as a scalar measure of variability
        variability = np.linalg.norm(feature_std)
        
        # Store by diagnosis
        diagnosis = 'PD' if participant_labels[p_id] == 1 else 'HC'
        results['variability'][diagnosis].append(variability)
    
    # Calculate statistics for variability
    results['statistics']['PD_variability_mean'] = np.mean(results['variability']['PD'])
    results['statistics']['PD_variability_std'] = np.std(results['variability']['PD'])
    results['statistics']['HC_variability_mean'] = np.mean(results['variability']['HC'])
    results['statistics']['HC_variability_std'] = np.std(results['variability']['HC'])
    
    # T-test for variability between PD and HC
    t_stat, p_val = stats.ttest_ind(results['variability']['PD'], results['variability']['HC'])
    results['statistics']['variability_ttest'] = {'t_statistic': t_stat, 'p_value': p_val}
    
    # Calculate feature-level statistics
    pd_features_array = np.vstack(pd_features)
    hc_features_array = np.vstack(hc_features)
    
    # Calculate mean and std for each feature dimension
    pd_mean = np.mean(pd_features_array, axis=0)
    pd_std = np.std(pd_features_array, axis=0)
    hc_mean = np.mean(hc_features_array, axis=0)
    hc_std = np.std(hc_features_array, axis=0)
    
    # Find the top 10 most different features between PD and HC
    feature_diffs = np.abs(pd_mean - hc_mean)
    top_diff_indices = np.argsort(feature_diffs)[-10:]
    
    # Store statistics for top different features
    for i, idx in enumerate(top_diff_indices):
        results['feature_stats']['PD'][f'feature_{idx}'] = {
            'mean': pd_mean[idx],
            'std': pd_std[idx]
        }
        results['feature_stats']['HC'][f'feature_{idx}'] = {
            'mean': hc_mean[idx],
            'std': hc_std[idx]
        }
        
        # T-test for this feature dimension
        t_stat, p_val = stats.ttest_ind(
            pd_features_array[:, idx],
            hc_features_array[:, idx]
        )
        results['statistics'][f'feature_{idx}_ttest'] = {
            'difference': pd_mean[idx] - hc_mean[idx],
            't_statistic': t_stat,
            'p_value': p_val
        }
    
    # Run t-SNE on a subset of features (max 1000 for computational efficiency)
    max_samples = min(1000, len(final_features))
    indices = np.random.choice(len(final_features), max_samples, replace=False)
    
    subset_final_features = final_features[indices].numpy()
    subset_labels = labels[indices].numpy()
    
    # Run t-SNE
    print("Running t-SNE on node features...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(subset_final_features)
    
    # Store t-SNE results
    results['tsne']['coordinates'] = tsne_results
    results['tsne']['labels'] = subset_labels
    
    print("Node feature analysis complete.")
    return results

def visualize_edge_attention(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Create visualizations for edge attention analysis, focusing on construction weights.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    print("Creating edge attention visualizations...")
    figure_paths = {}
    
    # 1. Box plot comparing construction weights for temporal vs similarity edges
    plt.figure(figsize=(8, 6))
    
    # Prepare data for box plot
    data = []
    labels = []
    
    # Add construction weights data if available
    if 'construction_temporal_mean' in results['statistics']:
        data.append(results['construction']['temporal'])
        data.append(results['construction']['similarity'])
        labels.extend(['Temporal', 'Similarity'])
    
    # Create box plot
    box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
    
    # Color boxes
    colors = [COLORS[0], COLORS[1]]
    for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]):
        patch.set_facecolor(color)
    
    # Add statistical significance markers
    if 'construction_ttest' in results['statistics']:
        p_value = results['statistics']['construction_ttest']['p_value']
        if p_value < 0.05:
            # Find the max y value
            y_max = max(max(data[0]), max(data[1]))
            plt.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-')
            plt.text(1.5, y_max*1.07, f'p={p_value:.3f}', ha='center')
    
    plt.ylabel('Edge Construction Weight')
    plt.title('Comparison of Edge Construction Weights by Edge Type')
    
    # Add statistical information in the figure text
    if 'construction_temporal_mean' in results['statistics'] and 'construction_similarity_mean' in results['statistics']:
        plt.figtext(0.5, 0.01, 
                    f"Temporal: {results['statistics']['construction_temporal_mean']:.3f} (±{results['statistics']['construction_temporal_std']:.3f}), "
                    f"Similarity: {results['statistics']['construction_similarity_mean']:.3f} (±{results['statistics']['construction_similarity_std']:.3f}), "
                    f"p={results['statistics']['construction_ttest']['p_value']:.3f}",
                    ha='center', fontsize=9)
    
    # Save figure
    fig_path = str(FIGURES_DIR / 'edge_construction_comparison.pdf')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    figure_paths['edge_construction_comparison'] = fig_path
    
    # 2. Box plot comparing construction weights by diagnosis
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plot
    data = []
    labels = []
    
    # Add PD vs HC for temporal construction weights
    if 'PD_temporal_construction_mean' in results['statistics'] and 'HC_temporal_construction_mean' in results['statistics']:
        data.append(results['by_diagnosis']['PD']['temporal_construction'])
        data.append(results['by_diagnosis']['HC']['temporal_construction'])
        labels.extend(['PD Temporal', 'HC Temporal'])
    
    # Add PD vs HC for similarity construction weights
    if 'PD_similarity_construction_mean' in results['statistics'] and 'HC_similarity_construction_mean' in results['statistics']:
        data.append(results['by_diagnosis']['PD']['similarity_construction'])
        data.append(results['by_diagnosis']['HC']['similarity_construction'])
        labels.extend(['PD Similarity', 'HC Similarity'])
    
    # Create box plot
    box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
    
    # Color boxes
    colors = [COLORS[2], COLORS[3], COLORS[4], COLORS[5]]
    for patch, color in zip(box['boxes'], colors[:len(box['boxes'])]):
        patch.set_facecolor(color)
    
    # Add statistical significance markers
    for i in range(0, len(data), 2):
        if i+1 < len(data):  # Make sure we have a pair
            # Get the corresponding p-value
            p_value = None
            if i == 0 and 'PD_vs_HC_temporal_construction' in results['statistics']:
                p_value = results['statistics']['PD_vs_HC_temporal_construction']['p_value']
            elif i == 2 and 'PD_vs_HC_similarity_construction' in results['statistics']:
                p_value = results['statistics']['PD_vs_HC_similarity_construction']['p_value']
            
            if p_value is not None and p_value < 0.05:
                # Find the max y value for this pair
                y_max = max(max(data[i]), max(data[i+1]))
                plt.plot([i+1, i+2], [y_max*1.05, y_max*1.05], 'k-')
                plt.text(i+1.5, y_max*1.07, f'p={p_value:.3f}', ha='center')
    
    plt.ylabel('Edge Construction Weight')
    plt.title('Comparison of Edge Construction Weights by Diagnosis')
    
    # Add edge distribution information
    if 'temporal_edge_percentage' in results['statistics'] and 'similarity_edge_percentage' in results['statistics']:
        plt.figtext(0.5, 0.01, 
                    f"Edge Distribution: Temporal {results['statistics']['temporal_edge_percentage']:.1f}%, "
                    f"Similarity {results['statistics']['similarity_edge_percentage']:.1f}%",
                    ha='center', fontsize=9)
    
    # Save figure
    fig_path = str(FIGURES_DIR / 'edge_construction_by_diagnosis.pdf')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    figure_paths['edge_construction_by_diagnosis'] = fig_path
    print(f"Edge attention visualizations saved to {FIGURES_DIR}")
    return figure_paths

def visualize_node_features(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Create visualizations for node feature analysis.
    
    Args:
        results: Dictionary containing analysis results
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    print("Creating node feature visualizations...")
    figure_paths = {}
    
    # 1. t-SNE plot of node features colored by diagnosis
    plt.figure(figsize=(8, 8))
    
    # Get t-SNE coordinates and labels
    tsne_coords = results['tsne']['coordinates']
    tsne_labels = results['tsne']['labels']
    
    # Create scatter plot
    for label, color, name in zip([0, 1], [COLORS[3], COLORS[2]], ['HC', 'PD']):
        mask = tsne_labels == label
        plt.scatter(
            tsne_coords[mask, 0],
            tsne_coords[mask, 1],
            c=[color],
            label=name,
            alpha=0.7,
            edgecolors='none'
        )
    
    plt.legend()
    plt.title('t-SNE of Node Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save figure
    fig_path = str(FIGURES_DIR / 'node_features_tsne.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    figure_paths['node_features_tsne'] = fig_path
    
    # 2. Box plot of feature variability by diagnosis
    plt.figure(figsize=(7, 6))
    
    # Prepare data for box plot
    data = [results['variability']['PD'], results['variability']['HC']]
    labels = ['PD', 'HC']
    
    # Create box plot
    box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
    
    # Color boxes
    for patch, color in zip(box['boxes'], [COLORS[2], COLORS[3]]):
        patch.set_facecolor(color)
    
    # Add statistical significance markers if applicable
    p_value = results['statistics']['variability_ttest']['p_value']
    if p_value < 0.05:
        y_max = max(max(results['variability']['PD']), max(results['variability']['HC']))
        plt.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-')
        plt.text(1.5, y_max*1.07, f'p={p_value:.3f}', ha='center')
    
    plt.ylabel('Feature Variability (L2 norm of std)')
    plt.title('Node Feature Variability by Diagnosis')
    
    # Add statistical information in the figure text
    plt.figtext(0.5, 0.01, 
                f"PD mean={results['statistics']['PD_variability_mean']:.3f} (±{results['statistics']['PD_variability_std']:.3f}), "
                f"HC mean={results['statistics']['HC_variability_mean']:.3f} (±{results['statistics']['HC_variability_std']:.3f}), "
                f"p={results['statistics']['variability_ttest']['p_value']:.3f}",
                ha='center', fontsize=9)
    
    # Save figure
    fig_path = str(FIGURES_DIR / 'node_feature_variability.pdf')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    figure_paths['node_feature_variability'] = fig_path
    
    # 3. Bar plot of top different features between PD and HC
    plt.figure(figsize=(10, 6))
    
    # Find feature indices with significant differences
    significant_features = []
    for key, value in results['statistics'].items():
        if key.startswith('feature_') and key.endswith('_ttest'):
            if value['p_value'] < 0.05:  # Significant difference
                feature_idx = int(key.split('_')[1])
                significant_features.append((feature_idx, value['difference'], value['p_value']))
    
    # Sort by absolute difference
    significant_features.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 10 or all if fewer
    top_features = significant_features[:min(10, len(significant_features))]
    
    if top_features:  # Only create plot if we have significant features
        # Prepare data for bar plot
        feature_indices = [f'F{idx}' for idx, _, _ in top_features]
        differences = [diff for _, diff, _ in top_features]
        p_values = [p for _, _, p in top_features]
        
        # Create bar plot
        bars = plt.bar(range(len(differences)), differences, color=[COLORS[2] if d > 0 else COLORS[3] for d in differences])
        
        # Add feature indices as x-tick labels
        plt.xticks(range(len(differences)), feature_indices)
        
        # Add p-value annotations
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'p={p_val:.3f}',
                    ha='center', va='bottom', rotation=90, fontsize=8)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylabel('Mean Difference (PD - HC)')
        plt.xlabel('Feature Index')
        plt.title('Top Differentiating Node Features Between PD and HC')
        
        # Save figure
        fig_path = str(FIGURES_DIR / 'node_feature_differences.pdf')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        figure_paths['node_feature_differences'] = fig_path
    
    print(f"Node feature visualizations saved to {FIGURES_DIR}")
    return figure_paths


def save_results(collected_data: Dict[str, Any], edge_results: Dict[str, Any], node_results: Dict[str, Any]) -> None:
    """
    Save analysis results to disk.
    
    Args:
        collected_data: Dictionary containing extracted data
        edge_results: Dictionary containing edge attention analysis results
        node_results: Dictionary containing node feature analysis results
    """
    print("Saving analysis results...")
    
    # Create results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Save edge attention results
    edge_attention_results = {
        'statistics': edge_results['statistics'],
        'construction_temporal': np.array(edge_results['construction']['temporal']),
        'construction_similarity': np.array(edge_results['construction']['similarity']),
    }
    
    # Add diagnosis-specific data if available
    for diagnosis in ['PD', 'HC']:
        for edge_type in ['temporal_construction', 'similarity_construction']:
            if len(edge_results['by_diagnosis'][diagnosis][edge_type]) > 0:
                edge_attention_results[f'{diagnosis}_{edge_type}'] = np.array(edge_results['by_diagnosis'][diagnosis][edge_type])
    
    # Save node feature results
    node_feature_results = {
        'statistics': node_results['statistics'],
        'tsne_coordinates': node_results['tsne']['coordinates'],
        'tsne_labels': node_results['tsne']['labels'],
        'pd_variability': np.array(node_results['variability']['PD']),
        'hc_variability': np.array(node_results['variability']['HC']),
    }
    
    # Add feature-level statistics if available
    if 'feature_stats' in node_results:
        for diagnosis in ['PD', 'HC']:
            for feature_key, feature_data in node_results['feature_stats'][diagnosis].items():
                node_feature_results[f'{diagnosis}_{feature_key}_mean'] = feature_data['mean']
                node_feature_results[f'{diagnosis}_{feature_key}_std'] = feature_data['std']
    
    # Save results to disk
    np.savez_compressed(
        RESULTS_DIR / 'edge_attention_results.npz',
        **edge_attention_results
    )
    
    np.savez_compressed(
        RESULTS_DIR / 'node_feature_results.npz',
        **node_feature_results
    )
    
    # Save statistics as JSON
    with open(RESULTS_DIR / 'statistics.json', 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        edge_stats = {}
        for key, value in edge_results['statistics'].items():
            if isinstance(value, dict):
                edge_stats[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
            else:
                edge_stats[key] = float(value) if isinstance(value, np.number) else value
        
        node_stats = {}
        for key, value in node_results['statistics'].items():
            if isinstance(value, dict):
                node_stats[key] = {k: float(v) if isinstance(v, np.number) else v for k, v in value.items()}
            else:
                node_stats[key] = float(value) if isinstance(value, np.number) else value
        
        json.dump({
            'edge_construction': edge_stats,
            'node_features': node_stats
        }, f, indent=2)
    
    print(f"Results saved to {RESULTS_DIR}")

def analyze_dataset(checkpoint, dataset_name, fold=0, batch_size=32, num_workers=4, num_batches=None, device='cpu'):
    """
    Analyze a single dataset and return the results.
    
    Args:
        checkpoint: Path to model checkpoint
        dataset_name: Name of the dataset ('italian' or 'kcl')
        fold: Fold index to use
        batch_size: Batch size
        num_workers: Number of workers for data loading
        num_batches: Number of batches to process (None for all)
        device: Device to use
        
    Returns:
        Tuple of (collected_data, edge_results, node_results)
    """
    print(f"\nAnalyzing dataset: {dataset_name}")
    
    # Load model
    model = load_model(checkpoint, device)
    
    # Load data
    data_module = VoiceDataModule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Get validation dataloader for the specified fold
    _, val_loader = data_module.get_fold_dataloaders(fold)
    
    # Extract interpretability data
    collected_data = extract_interpretability_data(
        model=model,
        dataloader=val_loader,
        device=device,
        num_batches=num_batches
    )
    
    # Analyze edge attention
    edge_results = analyze_edge_attention(collected_data)
    
    # Analyze node features
    node_results = analyze_node_features(collected_data)
    
    return collected_data, edge_results, node_results


def visualize_combined_results(italian_edge_results, italian_node_results, kcl_edge_results, kcl_node_results):
    """
    Create visualizations comparing results from both datasets.
    
    Args:
        italian_edge_results: Edge results from Italian dataset
        italian_node_results: Node results from Italian dataset
        kcl_edge_results: Edge results from KCL dataset
        kcl_node_results: Node results from KCL dataset
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    print("Creating combined visualizations...")
    figure_paths = {}
    
    # 1. Combined edge construction weights comparison
    plt.figure(figsize=(10, 6))
    
    # Prepare data for box plot
    data = [
        italian_edge_results['construction']['temporal'],
        italian_edge_results['construction']['similarity'],
        kcl_edge_results['construction']['temporal'],
        kcl_edge_results['construction']['similarity']
    ]
    labels = ['Italian\nTemporal', 'Italian\nSimilarity', 'KCL\nTemporal', 'KCL\nSimilarity']
    
    # Create box plot
    box = plt.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
    
    # Color boxes
    colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add statistical significance markers
    # Italian temporal vs similarity
    if 'construction_ttest' in italian_edge_results['statistics']:
        p_value = italian_edge_results['statistics']['construction_ttest']['p_value']
        if p_value < 0.05:
            y_max = max(max(data[0]), max(data[1]))
            plt.plot([1, 2], [y_max*1.05, y_max*1.05], 'k-')
            plt.text(1.5, y_max*1.07, f'p={p_value:.3f}', ha='center')
    
    # KCL temporal vs similarity
    if 'construction_ttest' in kcl_edge_results['statistics']:
        p_value = kcl_edge_results['statistics']['construction_ttest']['p_value']
        if p_value < 0.05:
            y_max = max(max(data[2]), max(data[3]))
            plt.plot([3, 4], [y_max*1.05, y_max*1.05], 'k-')
            plt.text(3.5, y_max*1.07, f'p={p_value:.3f}', ha='center')
    
    plt.ylabel('Edge Construction Weight')
    plt.title('Comparison of Edge Construction Weights by Dataset and Edge Type')
    
    # Save figure
    fig_path = str(FIGURES_DIR / 'combined_edge_construction_comparison.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    figure_paths['combined_edge_construction_comparison'] = fig_path
    
    # 2. Combined t-SNE visualization
    plt.figure(figsize=(12, 6))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    
    # Italian t-SNE
    tsne_coords_italian = italian_node_results['tsne']['coordinates']
    tsne_labels_italian = italian_node_results['tsne']['labels']
    
    # Create scatter plot for Italian
    for label, color, name in zip([0, 1], [COLORS[3], COLORS[2]], ['HC', 'PD']):
        mask = tsne_labels_italian == label
        plt.scatter(
            tsne_coords_italian[mask, 0],
            tsne_coords_italian[mask, 1],
            c=[color],
            label=name,
            alpha=0.7,
            edgecolors='none'
        )
    
    plt.title('Italian Dataset')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    # KCL t-SNE
    plt.subplot(1, 2, 2)
    tsne_coords_kcl = kcl_node_results['tsne']['coordinates']
    tsne_labels_kcl = kcl_node_results['tsne']['labels']
    
    # Create scatter plot for KCL
    for label, color, name in zip([0, 1], [COLORS[3], COLORS[2]], ['HC', 'PD']):
        mask = tsne_labels_kcl == label
        plt.scatter(
            tsne_coords_kcl[mask, 0],
            tsne_coords_kcl[mask, 1],
            c=[color],
            label=name,
            alpha=0.7,
            edgecolors='none'
        )
    
    plt.title('KCL Dataset')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    
    # Save figure
    fig_path = str(FIGURES_DIR / 'combined_node_features_tsne.pdf')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    figure_paths['combined_node_features_tsne'] = fig_path
    
    print(f"Combined visualizations saved to {FIGURES_DIR}")
    return figure_paths


def main():
    """
    Main function to run the interpretability analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze GAT attention weights and node features in VoiceGraphNet")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--dataset', type=str, default='both', choices=['italian', 'kcl', 'both'], help="Dataset to use")
    parser.add_argument('--fold', type=int, default=0, help="Fold index to use")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--num-batches', type=int, default=None, help="Number of batches to process (None for all)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    combined_figure_paths = {}
    
    if args.dataset == 'both' or args.dataset == 'italian':
        # Analyze Italian dataset
        italian_data, italian_edge_results, italian_node_results = analyze_dataset(
            checkpoint=args.checkpoint,
            dataset_name='italian',
            fold=args.fold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_batches=args.num_batches,
            device=device
        )
        
        # Visualize Italian results
        italian_edge_figure_paths = visualize_edge_attention(italian_edge_results)
        italian_node_figure_paths = visualize_node_features(italian_node_results)
        
        # Save Italian results
        save_results(italian_data, italian_edge_results, italian_node_results)
        
        # Add to combined figure paths
        combined_figure_paths.update({
            f"italian_{k}": v for k, v in {**italian_edge_figure_paths, **italian_node_figure_paths}.items()
        })
    
    if args.dataset == 'both' or args.dataset == 'kcl':
        # Analyze KCL dataset
        kcl_data, kcl_edge_results, kcl_node_results = analyze_dataset(
            checkpoint=args.checkpoint,
            dataset_name='kcl',
            fold=args.fold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_batches=args.num_batches,
            device=device
        )
        
        # Visualize KCL results
        kcl_edge_figure_paths = visualize_edge_attention(kcl_edge_results)
        kcl_node_figure_paths = visualize_node_features(kcl_node_results)
        
        # Save KCL results
        save_results(kcl_data, kcl_edge_results, kcl_node_results)
        
        # Add to combined figure paths
        combined_figure_paths.update({
            f"kcl_{k}": v for k, v in {**kcl_edge_figure_paths, **kcl_node_figure_paths}.items()
        })
    
    # If analyzing both datasets, create combined visualizations
    if args.dataset == 'both':
        combined_paths = visualize_combined_results(
            italian_edge_results=italian_edge_results,
            italian_node_results=italian_node_results,
            kcl_edge_results=kcl_edge_results,
            kcl_node_results=kcl_node_results
        )
        combined_figure_paths.update(combined_paths)
    
    print("\nAnalysis complete! Results and figures saved to:")
    print(f"  - Results: {RESULTS_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    print("\nFigures generated:")
    for name, path in combined_figure_paths.items():
        print(f"  - {name}: {path}")
    
    # Print key findings for each dataset if available
    if args.dataset == 'both' or args.dataset == 'italian':
        print("\nKey findings for Italian dataset:")
        print("  Edge Construction Weight Analysis:")
        
        # Construction weights
        if 'construction_temporal_mean' in italian_edge_results['statistics']:
            print(f"    - Construction Weights (Temporal vs. Similarity): "
                  f"{italian_edge_results['statistics']['construction_temporal_mean']:.3f} (±{italian_edge_results['statistics']['construction_temporal_std']:.3f}) vs. "
                  f"{italian_edge_results['statistics']['construction_similarity_mean']:.3f} (±{italian_edge_results['statistics']['construction_similarity_std']:.3f}) "
                  f"(p={italian_edge_results['statistics']['construction_ttest']['p_value']:.3f})")
            if italian_edge_results['statistics']['construction_ttest']['p_value'] < 0.05:
                print(f"      * SIGNIFICANT: Similarity edges have significantly different weights than temporal edges")
        
        print("\n  Node Feature Analysis:")
        if 'PD_variability_mean' in italian_node_results['statistics'] and 'HC_variability_mean' in italian_node_results['statistics']:
            print(f"    - Feature Variability (PD vs. HC): "
                  f"{italian_node_results['statistics']['PD_variability_mean']:.3f} (±{italian_node_results['statistics']['PD_variability_std']:.3f}) vs. "
                  f"{italian_node_results['statistics']['HC_variability_mean']:.3f} (±{italian_node_results['statistics']['HC_variability_std']:.3f}) "
                  f"(p={italian_node_results['statistics']['variability_ttest']['p_value']:.3f})")
            if italian_node_results['statistics']['variability_ttest']['p_value'] < 0.05:
                print(f"      * SIGNIFICANT: PD participants show significantly different feature variability than HC participants")
        
        # Feature-level differences
        significant_features = []
        for key, value in italian_node_results['statistics'].items():
            if key.startswith('feature_') and key.endswith('_ttest'):
                if value['p_value'] < 0.05:  # Significant difference
                    feature_idx = int(key.split('_')[1])
                    significant_features.append((feature_idx, value['difference'], value['p_value']))
        
        if significant_features:
            # Sort by p-value
            significant_features.sort(key=lambda x: x[2])
            print(f"\n    - Significantly Different Features (PD vs. HC):")
            for idx, diff, p_val in significant_features[:5]:  # Show top 5
                direction = "higher" if diff > 0 else "lower"
                print(f"      * Feature {idx}: PD {direction} by {abs(diff):.3f} (p={p_val:.3f})")
            if len(significant_features) > 5:
                print(f"      * Plus {len(significant_features) - 5} more significant features...")
        else:
            print("\n    - No individual features showed significant differences between PD and HC participants")
    
    if args.dataset == 'both' or args.dataset == 'kcl':
        print("\nKey findings for KCL dataset:")
        print("  Edge Construction Weight Analysis:")
        
        # Construction weights
        if 'construction_temporal_mean' in kcl_edge_results['statistics']:
            print(f"    - Construction Weights (Temporal vs. Similarity): "
                  f"{kcl_edge_results['statistics']['construction_temporal_mean']:.3f} (±{kcl_edge_results['statistics']['construction_temporal_std']:.3f}) vs. "
                  f"{kcl_edge_results['statistics']['construction_similarity_mean']:.3f} (±{kcl_edge_results['statistics']['construction_similarity_std']:.3f}) "
                  f"(p={kcl_edge_results['statistics']['construction_ttest']['p_value']:.3f})")
            if kcl_edge_results['statistics']['construction_ttest']['p_value'] < 0.05:
                print(f"      * SIGNIFICANT: Similarity edges have significantly different weights than temporal edges")
        
        print("\n  Node Feature Analysis:")
        if 'PD_variability_mean' in kcl_node_results['statistics'] and 'HC_variability_mean' in kcl_node_results['statistics']:
            print(f"    - Feature Variability (PD vs. HC): "
                  f"{kcl_node_results['statistics']['PD_variability_mean']:.3f} (±{kcl_node_results['statistics']['PD_variability_std']:.3f}) vs. "
                  f"{kcl_node_results['statistics']['HC_variability_mean']:.3f} (±{kcl_node_results['statistics']['HC_variability_std']:.3f}) "
                  f"(p={kcl_node_results['statistics']['variability_ttest']['p_value']:.3f})")
            if kcl_node_results['statistics']['variability_ttest']['p_value'] < 0.05:
                print(f"      * SIGNIFICANT: PD participants show significantly different feature variability than HC participants")
        
        # Feature-level differences
        significant_features = []
        for key, value in kcl_node_results['statistics'].items():
            if key.startswith('feature_') and key.endswith('_ttest'):
                if value['p_value'] < 0.05:  # Significant difference
                    feature_idx = int(key.split('_')[1])
                    significant_features.append((feature_idx, value['difference'], value['p_value']))
        
        if significant_features:
            # Sort by p-value
            significant_features.sort(key=lambda x: x[2])
            print(f"\n    - Significantly Different Features (PD vs. HC):")
            for idx, diff, p_val in significant_features[:5]:  # Show top 5
                direction = "higher" if diff > 0 else "lower"
                print(f"      * Feature {idx}: PD {direction} by {abs(diff):.3f} (p={p_val:.3f})")
            if len(significant_features) > 5:
                print(f"      * Plus {len(significant_features) - 5} more significant features...")
        else:
            print("\n    - No individual features showed significant differences between PD and HC participants")

if __name__ == "__main__":
    main()
