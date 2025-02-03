import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import networkx as nx
from scipy.signal import savgol_filter
import os
import argparse
from sklearn.manifold import TSNE
import seaborn as sns

from models.cnn_baseline import CNNBaseline
from models.lstm_baseline import LSTMBaseline
from models.transformer_baseline import TransformerBaseline
from models.voice_graph_net import VoiceGraphNet
from data.dataset import VoiceDataset, VoiceDataModule

def load_model(model_class, model_path):
    """Load a model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def get_dataset_samples(dataset_name):
    """Get representative samples from the actual dataset."""
    data = VoiceDataModule(dataset_name=dataset_name.lower(), segment_length=16000, batch_size=1)
    data.load_data()
    
    # Get validation data from the first fold
    splits = data.get_participant_splits()
    _, val_idx = splits[0]
    
    # Create validation dataset
    val_segments = data.segments[val_idx]
    val_metadata = data.metadata.iloc[val_idx].reset_index(drop=True)
    val_dataset = VoiceDataset(val_segments, val_metadata, segment_length=16000)
    
    # Create separate datasets for PD and Control samples
    pd_mask = val_metadata['clean_label'] == 1
    control_mask = val_metadata['clean_label'] == 0
    
    pd_dataset = VoiceDataset(val_segments[pd_mask], val_metadata[pd_mask].reset_index(drop=True), segment_length=16000)
    control_dataset = VoiceDataset(val_segments[control_mask], val_metadata[control_mask].reset_index(drop=True), segment_length=16000)
    
    # Get one sample from each
    pd_batch = next(iter(torch.utils.data.DataLoader(pd_dataset, batch_size=1, shuffle=True)))
    control_batch = next(iter(torch.utils.data.DataLoader(control_dataset, batch_size=1, shuffle=True)))
    
    return pd_batch, control_batch

def plot_cnn_importance(model, batch, title, ax, cmap, norm):
    """Plot CNN feature importance using activation analysis."""
    model.eval()
    with torch.no_grad():
        # Process through model
        x = batch['segment'].unsqueeze(1)  # Add channel dimension
        
        # Get intermediate activations
        activations = []
        def hook(module, input, output):
            activations.append(output.cpu().numpy())
        
        # Register hooks for convolutional layers
        hooks = []
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                hooks.append(module.register_forward_hook(hook))
        
        # Forward pass
        outputs = model(batch)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Get feature importance from last conv layer
        feature_importance = np.abs(activations[-1][0]).mean(axis=0)  # Average across channels
        
        # Normalize importance scores
        feature_importance = feature_importance - feature_importance.min()
        feature_importance = feature_importance / (feature_importance.max() + 1e-8)
        
        # Plot waveform and importance
        signal = batch['segment'][0].cpu().numpy()
        time = np.linspace(0, 1, len(signal))
        
        # Smooth signal
        window_length = max(3, int(len(signal) * 0.01) // 2 * 2 + 1)
        polyorder = min(3, window_length - 1)
        signal_smooth = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
        
        # Plot signal
        ax.plot(time, signal_smooth, 'k-', alpha=0.6, label='Waveform')
        
        # Plot importance as colored background
        importance_interp = np.interp(np.linspace(0, 1, len(signal)), 
                                    np.linspace(0, 1, len(feature_importance)), 
                                    feature_importance)
        
        ax.fill_between(time, signal_smooth, alpha=0.3, 
                       color=cmap(norm(importance_interp)))
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

def plot_lstm_attention(model, batch, title, ax, cmap, norm):
    """Plot LSTM attention weights."""
    model.eval()
    with torch.no_grad():
        # Process through model
        outputs = model(batch)
        attention = outputs['attention_weights'].cpu().numpy()
        
        # Ensure attention is 1D array
        if len(attention.shape) == 0:
            attention = np.array([attention])
        elif len(attention.shape) == 2:
            attention = attention.squeeze(0)  # Remove batch dimension
        
        # Plot waveform and attention
        signal = batch['segment'][0].cpu().numpy()
        time = np.linspace(0, 1, len(signal))
        
        # Smooth signal
        window_length = max(3, int(len(signal) * 0.01) // 2 * 2 + 1)
        polyorder = min(3, window_length - 1)
        signal_smooth = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
        
        # Plot signal
        ax.plot(time, signal_smooth, 'k-', alpha=0.6, label='Waveform')
        
        # Plot attention as colored background
        attention_interp = np.interp(np.linspace(0, 1, len(signal)), 
                                   np.linspace(0, 1, len(attention)), 
                                   attention)
        
        ax.fill_between(time, signal_smooth, alpha=0.3,
                       color=cmap(norm(attention_interp)))
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

def plot_transformer_attention(model, batch, title, ax, cmap, norm):
    """Plot Transformer attention weights."""
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        attention = outputs['attention_weights'][-1].cpu().numpy()
        attention = attention.mean(axis=1)[0, 0, :]  # Average across heads, get CLS token attention
        
        # Normalize attention
        attention = attention - attention.min()
        attention = attention / (attention.max() + 1e-8)
        
        # Plot waveform and attention
        signal = batch['segment'][0].cpu().numpy()
        time = np.linspace(0, 1, len(signal))
        
        # Smooth signal
        window_length = max(3, int(len(signal) * 0.01) // 2 * 2 + 1)
        polyorder = min(3, window_length - 1)
        signal_smooth = savgol_filter(signal, window_length=window_length, polyorder=polyorder)
        
        # Plot signal
        ax.plot(time, signal_smooth, 'k-', alpha=0.6, label='Waveform')
        
        # Plot attention as colored background
        attention_interp = np.interp(np.linspace(0, 1, len(signal)), 
                                   np.linspace(0, 1, len(attention)), 
                                   attention)
        
        ax.fill_between(time, signal_smooth, alpha=0.3,
                       color=cmap(norm(attention_interp)))
        
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')

def plot_graph_structure(ax, edge_index, edge_type, node_positions, attention_weights, title):
    """Plot the graph structure with attention weights."""
    # Create networkx graph
    G = nx.Graph()
    
    # Add nodes
    num_nodes = len(node_positions)
    G.add_nodes_from(range(num_nodes))
    
    # Normalize attention weights separately for temporal and similarity edges
    temporal_mask = edge_type == 0
    similarity_mask = edge_type == 1
    
    temporal_weights = attention_weights[temporal_mask]
    similarity_weights = attention_weights[similarity_mask]
    
    # Normalize each set of weights independently
    temporal_weights = (temporal_weights - temporal_weights.min()) / (temporal_weights.max() - temporal_weights.min() + 1e-8)
    similarity_weights = (similarity_weights - similarity_weights.min()) / (similarity_weights.max() - similarity_weights.min() + 1e-8)
    
    # Separate temporal and similarity edges
    temporal_edges = []
    temporal_edge_weights = []
    similarity_edges = []
    similarity_edge_weights = []
    
    temporal_idx = 0
    similarity_idx = 0
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        if edge_type[i] == 0:  # Temporal edge
            temporal_edges.append((src.item(), dst.item()))
            temporal_edge_weights.append(1.0 + 2.0 * temporal_weights[temporal_idx])
            temporal_idx += 1
        else:  # Similarity edge
            similarity_edges.append((src.item(), dst.item()))
            similarity_edge_weights.append(1.0 + 2.0 * similarity_weights[similarity_idx])
            similarity_idx += 1
    
    # Draw temporal edges
    nx.draw_networkx_edges(G, pos=node_positions,
                          edgelist=temporal_edges,
                          edge_color='blue',
                          width=temporal_edge_weights,
                          style='solid',
                          alpha=0.7,
                          ax=ax)
    
    # Draw similarity edges
    nx.draw_networkx_edges(G, pos=node_positions,
                          edgelist=similarity_edges,
                          edge_color='red',
                          width=similarity_edge_weights,
                          style='dashed',
                          alpha=0.5,
                          ax=ax)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos=node_positions,
                          node_color='lightgray',
                          node_size=500,
                          ax=ax)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos=node_positions,
                          font_size=8,
                          ax=ax)
    
    # Add legend
    temporal_line = plt.Line2D([], [], color='blue', linestyle='-', label='Temporal Edge')
    similarity_line = plt.Line2D([], [], color='red', linestyle='--', label='Similarity Edge')
    ax.legend(handles=[temporal_line, similarity_line])
    
    ax.set_title(title)
    ax.axis('equal')

def plot_attention_heatmap(ax, attention_matrix, edge_type, edge_index, title):
    """Plot attention weights as a heatmap."""
    # Get the maximum node index to determine matrix size
    n = edge_index.max() + 1
    temporal_matrix = np.zeros((n, n))
    similarity_matrix = np.zeros((n, n))
    
    # Fill matrices with attention values
    for i, att in enumerate(attention_matrix):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        if edge_type[i] == 0:  # Temporal attention
            temporal_matrix[src, dst] = 1.0  # Use fixed value for better visualization
            temporal_matrix[dst, src] = 1.0  # Make it symmetric
        else:  # Similarity attention
            similarity_matrix[src, dst] = 1.0  # Use fixed value for better visualization
            similarity_matrix[dst, src] = 1.0  # Make it symmetric
    
    # Create RGB matrix
    rgb_matrix = np.zeros((n, n, 3))
    rgb_matrix[:, :, 0] = temporal_matrix  # Red channel for temporal
    rgb_matrix[:, :, 2] = similarity_matrix  # Blue channel for similarity
    
    # Plot heatmap
    ax.imshow(rgb_matrix, aspect='auto')
    ax.set_title(title)
    ax.set_xlabel('Target Node')
    ax.set_ylabel('Source Node')
    
    # Add legend
    temporal_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5, label='Temporal Attention')
    similarity_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5, label='Similarity Attention')
    ax.legend(handles=[temporal_patch, similarity_patch], loc='upper right')

def get_node_positions(num_nodes):
    """Get node positions in a circular layout that emphasizes temporal sequence."""
    radius = 1.0
    angle_step = 2 * np.pi / num_nodes
    positions = {}
    
    for i in range(num_nodes):
        angle = i * angle_step
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        positions[i] = np.array([x, y])
    
    return positions

def plot_feature_evolution(ax, features_dict, title):
    """Plot how node features evolve through model stages."""
    # Get all features
    feature_stages = ['initial', 'domain', 'gat1', 'final']
    all_features = []
    stage_boundaries = [0]
    stage_centers = []
    
    # Normalize and collect all features with improved scaling
    for stage in feature_stages:
        features = features_dict[stage].detach().cpu().numpy()
        # Center and scale features using robust normalization
        features_mean = np.mean(features)
        features_std = np.std(features)
        features_norm = (features - features_mean) / (features_std + 1e-8)
        # Clip extreme values for better color distribution
        features_norm = np.clip(features_norm, -2, 2)
        # Scale to [0, 1] range
        features_norm = (features_norm + 2) / 4
        
        all_features.append(features_norm)
        stage_boundaries.append(stage_boundaries[-1] + features.shape[1])
        stage_centers.append((stage_boundaries[-2] + stage_boundaries[-1]) / 2)
    
    # Concatenate all features
    feature_evolution = np.concatenate(all_features, axis=1)
    
    # Create heatmap with improved colormap
    im = ax.imshow(feature_evolution, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    
    # Add vertical lines to separate stages
    for boundary in stage_boundaries[1:-1]:
        ax.axvline(x=boundary - 0.5, color='black', linestyle='-', alpha=0.3)
    
    # Customize plot with adjusted title position
    ax.set_title(title, pad=30, fontsize=12, fontweight='bold')
    ax.set_xlabel('Feature Dimension', labelpad=10)
    ax.set_ylabel('Node', labelpad=10)
    
    # Add stage labels with adjusted position
    for center, stage in zip(stage_centers, feature_stages):
        ax.text(center, -1, stage.upper(), ha='center', va='bottom', 
               rotation=0, fontsize=9, alpha=0.7)
    
    # Add colorbar with better labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature Intensity', labelpad=10)
    
    # Adjust layout
    ax.set_xticks([])  # Hide x-axis ticks since they're not meaningful
    ax.set_ylim(-2, feature_evolution.shape[0] + 0.5)

def generate_model_interpretability(model_class, model_path, pd_batch, control_batch, 
                                  save_dir, dataset_name, plot_func):
    """Generate interpretability visualization for a specific model."""
    # Load model
    model = load_model(model_class, model_path)
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), height_ratios=[1, 1])
    plt.subplots_adjust(hspace=0.3)
    
    # Create colormap
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Plot both panels
    plot_func(model, pd_batch, f'{dataset_name} PD', ax1, cmap, norm)
    plot_func(model, control_batch, f'{dataset_name} Control', ax2, cmap, norm)
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f'{model_class.__name__.lower()}_{dataset_name.lower()}.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def generate_graph_interpretability(model_path, pd_batch, control_batch, save_dir, dataset_name):
    """Generate graph model interpretability visualizations."""
    # Load model
    model = load_model(VoiceGraphNet, model_path)
    
    # Get model outputs
    with torch.no_grad():
        pd_outputs = model(pd_batch, return_attention=True)
        control_outputs = model(control_batch, return_attention=True)
    
    # Graph Structure Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
    
    # Plot graph structures
    pd_info = pd_outputs['attention_info']
    control_info = control_outputs['attention_info']
    
    plot_graph_structure(ax1, pd_info['edge_index'], pd_info['edge_type'],
                        get_node_positions(len(torch.unique(pd_info['edge_index']))),
                        pd_info['gat2_attention'], f'{dataset_name} PD')
    
    plot_graph_structure(ax2, control_info['edge_index'], control_info['edge_type'],
                        get_node_positions(len(torch.unique(control_info['edge_index']))),
                        control_info['gat2_attention'], f'{dataset_name} Control')
    
    # Plot attention heatmaps
    plot_attention_heatmap(ax3, pd_info['gat2_attention'], pd_info['edge_type'],
                          pd_info['edge_index'], f'{dataset_name} PD Attention')
    
    plot_attention_heatmap(ax4, control_info['gat2_attention'], control_info['edge_type'],
                          control_info['edge_index'], f'{dataset_name} Control Attention')
    
    plt.savefig(os.path.join(save_dir, f'graph_structure_{dataset_name.lower()}.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Feature Evolution Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    pd_features = model.get_all_intermediate_features(pd_batch)
    control_features = model.get_all_intermediate_features(control_batch)
    
    plot_feature_evolution(ax1, pd_features, f'{dataset_name} PD')
    plot_feature_evolution(ax2, control_features, f'{dataset_name} Control')
    
    plt.savefig(os.path.join(save_dir, f'feature_evolution_{dataset_name.lower()}.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def generate_combined_visualization(model_class, model_path, save_dir, plot_func, filename_prefix):
    """Generate combined visualization for both datasets."""
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    # Create colormap
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Process each dataset
    for idx, dataset_name in enumerate(['KCL', 'ITALIAN']):
        # Get dataset samples
        pd_batch, control_batch = get_dataset_samples(dataset_name)
        
        # Load model
        model = load_model(model_class, model_path)
        
        # Plot PD
        ax = fig.add_subplot(gs[idx, 0])
        plot_func(model, pd_batch, f'{dataset_name} PD', ax, cmap, norm)
        
        # Plot Control
        ax = fig.add_subplot(gs[idx, 1])
        plot_func(model, control_batch, f'{dataset_name} Control', ax, cmap, norm)
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f'{filename_prefix}_combined.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def generate_graph_combined_visualizations(model_path, save_dir):
    """Generate all combined visualizations for graph model."""
    # Load model
    model = load_model(VoiceGraphNet, model_path)
    
    # Get samples and outputs for all datasets
    outputs = {}
    for dataset_name in ['KCL', 'ITALIAN']:
        pd_batch, control_batch = get_dataset_samples(dataset_name)
        with torch.no_grad():
            outputs[dataset_name] = {
                'pd': model(pd_batch, return_attention=True),
                'control': model(control_batch, return_attention=True)
            }
    
    # Combined Graph Structure Visualization
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    for i, dataset_name in enumerate(['KCL', 'ITALIAN']):
        # PD
        ax = fig.add_subplot(gs[i, 0])
        info = outputs[dataset_name]['pd']['attention_info']
        plot_graph_structure(ax, info['edge_index'], info['edge_type'],
                           get_node_positions(len(torch.unique(info['edge_index']))),
                           info['gat2_attention'], f'{dataset_name} PD')
        
        # Control
        ax = fig.add_subplot(gs[i, 1])
        info = outputs[dataset_name]['control']['attention_info']
        plot_graph_structure(ax, info['edge_index'], info['edge_type'],
                           get_node_positions(len(torch.unique(info['edge_index']))),
                           info['gat2_attention'], f'{dataset_name} Control')
    
    plt.savefig(os.path.join(save_dir, 'combined_graph_structures.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Combined Attention Pattern Visualization
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    for i, dataset_name in enumerate(['KCL', 'ITALIAN']):
        # PD
        ax = fig.add_subplot(gs[i, 0])
        info = outputs[dataset_name]['pd']['attention_info']
        plot_attention_heatmap(ax, info['gat2_attention'], info['edge_type'],
                             info['edge_index'], f'{dataset_name} PD')
        
        # Control
        ax = fig.add_subplot(gs[i, 1])
        info = outputs[dataset_name]['control']['attention_info']
        plot_attention_heatmap(ax, info['gat2_attention'], info['edge_type'],
                             info['edge_index'], f'{dataset_name} Control')
    
    plt.savefig(os.path.join(save_dir, 'combined_attention_patterns.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Feature Space Visualization
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    def plot_features(ax, features, edge_index, edge_type, title):
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features.detach().cpu().numpy())
        
        # Create scatter plot
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                           c=np.arange(len(features_2d)), 
                           cmap='viridis',
                           s=100)
        
        # Draw edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            if edge_type[i] == 0:  # Temporal edge
                ax.plot([features_2d[src, 0], features_2d[dst, 0]],
                       [features_2d[src, 1], features_2d[dst, 1]],
                       'b-', alpha=0.2)
            else:  # Similarity edge
                ax.plot([features_2d[src, 0], features_2d[dst, 0]],
                       [features_2d[src, 1], features_2d[dst, 1]],
                       'r--', alpha=0.2)
        
        ax.set_title(title)
        plt.colorbar(scatter, ax=ax, label='Temporal Position')
    
    for i, dataset_name in enumerate(['KCL', 'ITALIAN']):
        # PD
        ax = fig.add_subplot(gs[i, 0])
        info = outputs[dataset_name]['pd']['attention_info']
        plot_features(ax, outputs[dataset_name]['pd']['features'],
                     info['edge_index'], info['edge_type'],
                     f'{dataset_name} PD')
        
        # Control
        ax = fig.add_subplot(gs[i, 1])
        info = outputs[dataset_name]['control']['attention_info']
        plot_features(ax, outputs[dataset_name]['control']['features'],
                     info['edge_index'], info['edge_type'],
                     f'{dataset_name} Control')
    
    plt.savefig(os.path.join(save_dir, 'feature_space_visualization.png'),
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    """Generate all interpretability visualizations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_model', type=str, required=True, help='Path to CNN model checkpoint')
    parser.add_argument('--lstm_model', type=str, required=True, help='Path to LSTM model checkpoint')
    parser.add_argument('--transformer_model', type=str, required=True, help='Path to Transformer model checkpoint')
    parser.add_argument('--graph_model', type=str, required=True, help='Path to Graph model checkpoint')
    parser.add_argument('--save_dir', type=str, default='visualization_results/figures')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set style for publication-quality figures
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Process each dataset
    for dataset_name in ['KCL', 'ITALIAN']:
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Get dataset samples
        pd_batch, control_batch = get_dataset_samples(dataset_name)
        
        # Generate CNN interpretability
        print("Generating CNN interpretability...")
        generate_model_interpretability(CNNBaseline, args.cnn_model,
                                     pd_batch, control_batch,
                                     args.save_dir, dataset_name,
                                     plot_cnn_importance)
        
        # Generate LSTM interpretability
        print("Generating LSTM interpretability...")
        generate_model_interpretability(LSTMBaseline, args.lstm_model,
                                     pd_batch, control_batch,
                                     args.save_dir, dataset_name,
                                     plot_lstm_attention)
        
        # Generate Transformer interpretability
        print("Generating Transformer interpretability...")
        generate_model_interpretability(TransformerBaseline, args.transformer_model,
                                     pd_batch, control_batch,
                                     args.save_dir, dataset_name,
                                     plot_transformer_attention)
        
        # Generate Graph interpretability
        print("Generating Graph interpretability...")
        generate_graph_interpretability(args.graph_model,
                                     pd_batch, control_batch,
                                     args.save_dir, dataset_name)
    
    # Generate combined visualizations
    print("\nGenerating combined visualizations...")
    
    # CNN combined
    generate_combined_visualization(CNNBaseline, args.cnn_model,
                                 args.save_dir, plot_cnn_importance,
                                 'cnn_analysis')
    
    # LSTM combined
    generate_combined_visualization(LSTMBaseline, args.lstm_model,
                                 args.save_dir, plot_lstm_attention,
                                 'lstm_analysis')
    
    # Transformer combined
    generate_combined_visualization(TransformerBaseline, args.transformer_model,
                                 args.save_dir, plot_transformer_attention,
                                 'transformer_analysis')
    
    # Graph combined visualizations
    generate_graph_combined_visualizations(args.graph_model, args.save_dir)
    
    print("\nAll interpretability visualizations generated successfully!")

if __name__ == '__main__':
    main() 