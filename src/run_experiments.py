import subprocess
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def run_command(cmd: List[str], desc: str = "") -> None:
    """Run a command and print its output."""
    print(f"\n{'='*80}\nRunning: {desc}\nCommand: {' '.join(cmd)}\n{'='*80}\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"\nError in command: {' '.join(cmd)}")
        for line in process.stderr:
            print(line, end='')
        return False
    return True

def train_all_models(
    save_dir: str = 'results',
    epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cuda',
    debug: bool = False,
    ultra_debug: bool = False
) -> None:
    """Train all models on both datasets."""
    if ultra_debug:
        print("\nRunning in ULTRA-DEBUG mode:")
        print("- Using 10 participants (5 PD, 5 Control)")
        print("- Single fold validation")
        print("- Single epoch training")
        epochs = 1
    
    models = [
        'cnn_baseline',
        'lstm_baseline',
        'transformer_baseline',
        'voice_graph_net'
    ]
    
    datasets = ['kcl', 'italian']
    
    # Create progress bars
    model_pbar = tqdm(models, desc='Models', position=0)
    dataset_pbar = tqdm(datasets, desc='Datasets', position=1, leave=False)
    
    for model in model_pbar:
        model_pbar.set_description(f"Training {model}")
        for dataset in dataset_pbar:
            dataset_pbar.set_description(f"Dataset: {dataset}")
            
            cmd = [
                'python', 'src/train.py',
                '--model', model,
                '--dataset', dataset,
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--device', device,
                '--save-dir', save_dir
            ]
            
            if debug:
                cmd.append('--debug')
            if ultra_debug:
                cmd.append('--ultra-debug')
            
            success = run_command(cmd, f"Training {model} on {dataset}")
            if not success:
                print(f"\nError training {model} on {dataset}. Continuing with next...")

def run_cross_dataset_validation(
    save_dir: str = 'results',
    batch_size: int = 32,
    device: str = 'cuda',
    debug: bool = False,
    ultra_debug: bool = False
) -> None:
    """Run cross-dataset validation for all models."""
    if ultra_debug:
        print("\nRunning cross-dataset validation in ULTRA-DEBUG mode")
    
    models = [
        'cnn_baseline',
        'lstm_baseline',
        'transformer_baseline',
        'voice_graph_net'
    ]
    
    # Create progress bar
    pbar = tqdm(total=len(models)*2, desc='Cross-dataset Validation')
    
    for model in models:
        # KCL → Italian
        pbar.set_description(f"Testing {model} (KCL → Italian)")
        checkpoint = f"{save_dir}/checkpoints/{model}/kcl/fold_0/best_model.pt"
        cmd = [
            'python', 'src/train.py',
            '--model', model,
            '--cross-dataset',
            '--train-dataset', 'kcl',
            '--test-dataset', 'italian',
            '--checkpoint', checkpoint,
            '--batch-size', str(batch_size),
            '--device', device,
            '--save-dir', save_dir
        ]
        if debug or ultra_debug:
            cmd.append('--debug')
        
        success = run_command(cmd, f"Cross-validation: {model} (KCL → Italian)")
        pbar.update(1)
        
        if not success:
            print(f"\nError in cross-validation for {model} (KCL → Italian). Continuing...")
            continue
        
        # Italian → KCL
        pbar.set_description(f"Testing {model} (Italian → KCL)")
        checkpoint = f"{save_dir}/checkpoints/{model}/italian/fold_0/best_model.pt"
        cmd = [
            'python', 'src/train.py',
            '--model', model,
            '--cross-dataset',
            '--train-dataset', 'italian',
            '--test-dataset', 'kcl',
            '--checkpoint', checkpoint,
            '--batch-size', str(batch_size),
            '--device', device,
            '--save-dir', save_dir
        ]
        if debug or ultra_debug:
            cmd.append('--debug')
        
        success = run_command(cmd, f"Cross-validation: {model} (Italian → KCL)")
        if not success:
            print(f"\nError in cross-validation for {model} (Italian → KCL). Continuing...")
        
        pbar.update(1)
    
    pbar.close()

def run_ablation_studies(
    save_dir: str = 'results',
    epochs: int = 50,
    batch_size: int = 32,
    device: str = 'cuda',
    debug: bool = False,
    ultra_debug: bool = False
) -> None:
    """Run ablation studies for the graph model."""
    if ultra_debug:
        print("\nRunning ablation studies in ULTRA-DEBUG mode")
        epochs = 1
    
    # Test different edge configurations
    edge_configs = [
        ('temporal_only', {'k_temporal': 5, 'k_similarity': 0}),
        ('similarity_only', {'k_temporal': 0, 'k_similarity': 5}),
        ('both', {'k_temporal': 5, 'k_similarity': 5})
    ]
    
    # Create progress bars
    config_pbar = tqdm(edge_configs, desc='Edge Configurations', position=0)
    dataset_pbar = tqdm(['kcl', 'italian'], desc='Datasets', position=1, leave=False)
    
    for name, config in config_pbar:
        config_pbar.set_description(f"Testing edge config: {name}")
        for dataset in dataset_pbar:
            dataset_pbar.set_description(f"Dataset: {dataset}")
            cmd = [
                'python', 'src/train.py',
                '--model', 'voice_graph_net',
                '--dataset', dataset,
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--device', device,
                '--save-dir', f"{save_dir}/ablation/edges/{name}"
            ]
            # Add edge configuration
            for param, value in config.items():
                cmd.extend([f'--{param}', str(value)])
            
            if debug:
                cmd.append('--debug')
            if ultra_debug:
                cmd.append('--ultra-debug')
            
            success = run_command(cmd, f"Ablation (edges): {name} on {dataset}")
            if not success:
                print(f"\nError in ablation study for edge config {name} on {dataset}. Continuing...")

def plot_comparison_results(save_dir: str = 'results') -> None:
    """Plot comparison of all models and datasets."""
    print("\nPlotting model comparison results...")
    save_dir = Path(save_dir)
    models = [
        'cnn_baseline',
        'lstm_baseline',
        'transformer_baseline',
        'voice_graph_net'
    ]
    datasets = ['kcl', 'italian']
    metrics = ['participant_auc', 'participant_f1', 'participant_accuracy']
    
    # Collect results
    results = {}
    for model in tqdm(models, desc='Loading results'):
        results[model] = {}
        for dataset in datasets:
            with open(save_dir / model / dataset / 'cv_results.json', 'r') as f:
                results[model][dataset] = json.load(f)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        data = []
        for model in models:
            for dataset in datasets:
                mean = results[model][dataset]['cv_means'][metric]
                std = results[model][dataset]['cv_stds'][metric]
                data.append({
                    'Model': model,
                    'Dataset': dataset,
                    'Mean': mean,
                    'Std': std
                })
        
        df = pd.DataFrame(data)
        sns.barplot(
            data=df,
            x='Model',
            y='Mean',
            hue='Dataset',
            ax=axes[i]
        )
        axes[i].set_title(metric.replace('participant_', '').upper())
        axes[i].set_ylim(0.5, 1.0)
        
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png')
    plt.close()

def plot_cross_dataset_results(save_dir: str = 'results') -> None:
    """Plot cross-dataset validation results."""
    print("\nPlotting cross-dataset validation results...")
    save_dir = Path(save_dir)
    models = [
        'cnn_baseline',
        'lstm_baseline',
        'transformer_baseline',
        'voice_graph_net'
    ]
    directions = [
        ('kcl', 'italian'),
        ('italian', 'kcl')
    ]
    metrics = ['participant_auc', 'participant_f1', 'participant_accuracy']
    
    # Collect results
    results = {}
    for model in tqdm(models, desc='Loading results'):
        results[model] = {}
        for train_dataset, test_dataset in directions:
            result_path = save_dir / model / f'{train_dataset}_to_{test_dataset}' / 'cross_dataset_results.json'
            with open(result_path, 'r') as f:
                results[model][f'{train_dataset}→{test_dataset}'] = json.load(f)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        data = []
        for model in models:
            for direction in [f'kcl→italian', f'italian→kcl']:
                value = results[model][direction]['metrics'][metric]
                data.append({
                    'Model': model,
                    'Direction': direction,
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        sns.barplot(
            data=df,
            x='Model',
            y='Value',
            hue='Direction',
            ax=axes[i]
        )
        axes[i].set_title(metric.replace('participant_', '').upper())
        axes[i].set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cross_dataset_results.png')
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default='results', help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--ultra-debug', action='store_true', help='Run in ultra-debug mode (1 fold, 1 epoch)')
    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting experiment pipeline...")
    print(f"Save directory: {save_dir}")
    print(f"Device: {args.device}")
    print(f"Debug mode: {'Ultra-Debug' if args.ultra_debug else 'Debug' if args.debug else 'Off'}")

    try:
        # Train all models
        train_all_models(
            save_dir=str(save_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            debug=args.debug,
            ultra_debug=args.ultra_debug
        )

        # Run cross-dataset validation
        run_cross_dataset_validation(
            save_dir=str(save_dir),
            batch_size=args.batch_size,
            device=args.device,
            debug=args.debug,
            ultra_debug=args.ultra_debug
        )

        # Run ablation studies
        run_ablation_studies(
            save_dir=str(save_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            debug=args.debug,
            ultra_debug=args.ultra_debug
        )

        # Plot results
        plot_comparison_results(str(save_dir))
        plot_cross_dataset_results(str(save_dir))

        print("\nExperiment pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in experiment pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main() 