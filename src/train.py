import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
import logging
from tqdm import tqdm
import time  # Add this import at the top
import random
from datetime import datetime
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import scipy.stats as t

from data.dataset import VoiceDataModule
from models.cnn_baseline import CNNBaseline
from models.lstm_baseline import LSTMBaseline
from models.transformer_baseline import TransformerBaseline
from models.voice_graph_net import VoiceGraphNet
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_reliability_diagram,
    plot_embedding_space,
    plot_model_comparison,
    print_metrics_with_ci
)

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = len(train_loader)
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            loss_dict = model.training_step(batch)
            loss = loss_dict['loss']
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'train_loss': total_loss / n_batches}

def calculate_bootstrap_ci(values, n_bootstrap=200, confidence=0.95, metric='accuracy'):
    """Calculate confidence interval using bootstrap.
    
    Args:
        values: List of values to bootstrap. For accuracy, these are individual values.
               For AUC, these should be tuples of (probability, label).
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default: 0.95 for 95% CI)
        metric: Either 'accuracy' or 'auc'
    
    Returns:
        Tuple of (lower_ci, upper_ci)
    """
    if len(values) < 2:
        return None, None
    
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(values), size=len(values), replace=True)
        
        if metric == 'auc':
            # For AUC, values should be tuples of (probability, label)
            bootstrap_probs = np.array([values[i][0] for i in indices])
            bootstrap_labels = np.array([values[i][1] for i in indices])
            
            # Skip if all labels are the same class
            if len(np.unique(bootstrap_labels)) < 2:
                continue
                
            try:
                auc = roc_auc_score(bootstrap_labels, bootstrap_probs)
                bootstrap_stats.append(auc)
            except ValueError:
                continue
        else:
            # For accuracy and other metrics, just take the mean
            bootstrap_sample = [values[i] for i in indices]
            bootstrap_stats.append(np.mean(bootstrap_sample))
    
    if len(bootstrap_stats) < n_bootstrap * 0.5:  # If we have too few valid bootstrap samples
        return None, None
    
    # Calculate percentile confidence interval
    lower_percentile = (1 - confidence) / 2
    upper_percentile = 1 - lower_percentile
    ci_lower = np.percentile(bootstrap_stats, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile * 100)
    
    return ci_lower, ci_upper

def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = len(val_loader)
    
    # Initialize lists to store predictions and labels
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_features = []
    participant_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            loss_dict = model.validation_step(batch)
            total_loss += loss_dict['val_loss'].item()
            
            # Store predictions and labels
            probs = torch.softmax(outputs['logits'], dim=1)
            preds = torch.argmax(outputs['logits'], dim=1)
            
            all_labels.append(batch['label'].cpu())
            all_predictions.append(preds.cpu())
            all_probabilities.append(probs[:, 1].cpu())
            
            if 'features' in outputs:
                all_features.append(outputs['features'].cpu())
    
    # Call on_validation_end if available
    if hasattr(model, 'on_validation_end'):
            model.on_validation_end()
    else:
        # Concatenate all tensors
        model.all_labels = torch.cat(all_labels)
        model.all_predictions = torch.cat(all_predictions)
        model.all_probabilities = torch.cat(all_probabilities)
        
        if all_features:
            model.all_features = torch.cat(all_features)
    
    # Save predictions for later analysis
    predictions = {
        'true_labels': model.all_labels.tolist(),
        'predictions': model.all_predictions.tolist(),
        'probabilities': model.all_probabilities.tolist(),
        'predicted_probabilities': model.all_probabilities.tolist()  # Add this for compatibility
    }
    
    # Calculate metrics
    correct = (model.all_predictions == model.all_labels).sum().item()
    total = len(model.all_labels)
    accuracy = correct / total
    auc = roc_auc_score(model.all_labels.cpu().numpy(), model.all_probabilities.cpu().numpy())
    
    # Calculate bootstrap CIs for segment-level metrics
    segment_acc_values = [(pred == label) for pred, label in zip(model.all_predictions.tolist(), model.all_labels.tolist())]
    segment_auc_values = [(prob, label) for prob, label in zip(model.all_probabilities.tolist(), model.all_labels.tolist())]
    
    acc_ci = calculate_bootstrap_ci(segment_acc_values, n_bootstrap=200)
    auc_ci = calculate_bootstrap_ci(segment_auc_values, n_bootstrap=200, metric='auc')
    
    # Initialize metrics dictionary with CIs
    metrics = {
        'val_loss': total_loss / n_batches,
        'accuracy': accuracy,
        'accuracy_ci': list(acc_ci) if acc_ci[0] is not None else None,
        'auc': auc,
        'auc_ci': list(auc_ci) if auc_ci[0] is not None else None
    }
    
    # Print metrics with CIs
    print(f"\nSegment-level Metrics:")
    print(f"Accuracy: {accuracy:.3f} [95% CI: {acc_ci[0]:.3f}-{acc_ci[1]:.3f}]")
    print(f"AUC: {auc:.3f} [95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f}]")
    
    # Compute participant-level metrics
    participant_results = model.compute_participant_metrics()
    metrics.update(participant_results)
    
    # Extract participant-level values for CI calculation
    acc_values = []
    auc_values = []
    if 'participant_metrics' in participant_results:
        for p_metrics in participant_results['participant_metrics']:
            if isinstance(p_metrics, dict):
                if 'accuracy' in p_metrics and isinstance(p_metrics['accuracy'], (int, float)) and not np.isnan(p_metrics['accuracy']):
                    acc_values.append(p_metrics['accuracy'])
                if 'auc' in p_metrics and isinstance(p_metrics['auc'], (int, float)) and not np.isnan(p_metrics['auc']):
                    auc_values.append(p_metrics['auc'])
    
    # Calculate bootstrap CIs (always do this, even in debug mode)
    if len(acc_values) >= 2 and len(auc_values) >= 2:
        acc_ci = calculate_bootstrap_ci(acc_values, n_bootstrap=200)
        auc_ci = calculate_bootstrap_ci(auc_values, n_bootstrap=200)
        
        # Store CIs and raw values in metrics
        metrics['participant_accuracy_ci'] = list(acc_ci) if acc_ci[0] is not None else None
        metrics['participant_auc_ci'] = list(auc_ci) if auc_ci[0] is not None else None
        metrics['participant_accuracy_values'] = acc_values
        metrics['participant_auc_values'] = auc_values
        metrics['participant_count'] = len(acc_values)
        
        # Also calculate segment-level CIs
        segment_acc_ci = calculate_bootstrap_ci(
            [(pred == label) for pred, label in zip(model.all_predictions.tolist(), model.all_labels.tolist())],
            n_bootstrap=200
        )
        segment_auc_ci = calculate_bootstrap_ci(
            [(prob, label) for prob, label in zip(model.all_probabilities.tolist(), model.all_labels.tolist())],
            n_bootstrap=200,
            metric='auc'
        )
        
        metrics['segment_accuracy_ci'] = list(segment_acc_ci) if segment_acc_ci[0] is not None else None
        metrics['segment_auc_ci'] = list(segment_auc_ci) if segment_auc_ci[0] is not None else None
        
        print(f"\nValidation Metrics (n={len(acc_values)} participants):")
        print(f"Participant Accuracy: {metrics['participant_accuracy']:.3f} [95% CI: {acc_ci[0]:.3f}-{acc_ci[1]:.3f}]")
        print(f"Participant AUC: {metrics['participant_auc']:.3f} [95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f}]")
        print(f"Segment Accuracy: {metrics['accuracy']:.3f} [95% CI: {segment_acc_ci[0]:.3f}-{segment_acc_ci[1]:.3f}]")
        print(f"Segment AUC: {metrics['auc']:.3f} [95% CI: {segment_auc_ci[0]:.3f}-{segment_auc_ci[1]:.3f}]")
    
    return metrics, predictions

def get_model(
    model_name: str,
    device: torch.device,
    **kwargs
) -> torch.nn.Module:
    """Get model instance by name."""
    if model_name == 'cnn_baseline':
        return CNNBaseline().to(device)
    elif model_name == 'lstm_baseline':
        return LSTMBaseline().to(device)
    elif model_name == 'transformer_baseline':
        return TransformerBaseline().to(device)
    elif model_name == 'voice_graph_net':
        return VoiceGraphNet().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(
    model_name: str,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    run_dir: Path = None,
    debug: bool = False,
    ultra_debug: bool = False,
    k_temporal: int = 5,
    k_similarity: int = 5,
    epochs: int = 1,
    **kwargs
) -> Tuple[Dict[str, float], Dict[str, List[float]], torch.nn.Module]:
    """Train a model on a dataset."""
    device = torch.device(device)
    
    # Load data
    data = VoiceDataModule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        ultra_debug=ultra_debug
    )
    train_loader, val_loader = data.get_fold_dataloaders(0)
    
    model = get_model(model_name, device)
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Lower learning rate for longer training
    
    # Add scheduler only for Graph Net
    scheduler = None
    if model_name == 'voice_graph_net':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    best_predictions = None
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        # Train
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            optimizer.zero_grad()
            loss_dict = model.training_step(batch)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        if scheduler is not None:
            scheduler.step()
        
        # Validate
        metrics, predictions = validate(model, val_loader, device)
        val_loss = metrics['val_loss']
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            best_predictions = predictions
            
            # Save model checkpoint
            if run_dir is not None:
                model_dir = run_dir / model_name / dataset_name
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / 'model.pt')
        
        # Save results
    if run_dir is not None:
        results_dir = run_dir / model_name / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(results_dir / "results.json", 'w') as f:
            json.dump({
                'model_name': model_name,
                'dataset_name': dataset_name,
                'metrics': best_metrics
            }, f, indent=4)
        
        # Save predictions
        with open(results_dir / "predictions.json", 'w') as f:
            json.dump(best_predictions, f, indent=4)
        
        # Generate plots
        plots_dir = results_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(
            model.all_labels.cpu().numpy(),
            model.all_predictions.cpu().numpy(),
            save_path=plots_dir / 'confusion_matrix.png'
        )
        
        # Reliability diagram
        plot_reliability_diagram(
            model.all_labels.cpu().numpy(),
            model.all_probabilities.cpu().numpy(),
            save_path=plots_dir / 'reliability_diagram.png'
        )
        
        # Feature space visualization if available
        if hasattr(model, 'all_features'):
            plot_embedding_space(
                model.all_features.cpu().numpy(),
                model.all_labels.cpu().numpy(),
                save_path=plots_dir / 'embedding_space.png'
            )
    
    return best_metrics, best_predictions, model

def cross_dataset_validation(
    model_name: str,
    train_dataset: str,
    test_dataset: str,
    checkpoint_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    run_dir: Path = None,
    debug: bool = False,
    ultra_debug: bool = False,
    k_temporal: int = 5,
    k_similarity: int = 5
) -> Dict[str, float]:
    """Evaluate a model trained on one dataset on another dataset."""
    device = torch.device(device)
    
    # Load test data
    test_data = VoiceDataModule(
        dataset_name=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        ultra_debug=ultra_debug
    )
    test_loader = test_data.get_fold_dataloaders(0)[1]  # Use validation loader
    
    # Load model with weights_only=True to avoid warning
    model = get_model(model_name, device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    # Evaluate
    metrics, predictions = validate(model, test_loader, device)
    
    # Save results
    results = {
        'model_name': model_name,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'metrics': metrics
    }
    
    # Save in cross_dataset_analysis directory
    results_dir = run_dir / model_name / f'{train_dataset}_to_{test_dataset}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save predictions
    with open(results_dir / "predictions.json", 'w') as f:
        json.dump(predictions, f, indent=4)
    
    # Generate plots
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    plot_confusion_matrix(
        model.all_labels.cpu().numpy(),
        model.all_predictions.cpu().numpy(),
        save_path=plots_dir / 'confusion_matrix.png'
    )
    
    # Reliability diagram
    plot_reliability_diagram(
        model.all_labels.cpu().numpy(),
        model.all_probabilities.cpu().numpy(),
        save_path=plots_dir / 'reliability_diagram.png'
    )
    
    # Feature space visualization if available
    if hasattr(model, 'all_features'):
        plot_embedding_space(
            model.all_features.cpu().numpy(),
            model.all_labels.cpu().numpy(),
            save_path=plots_dir / 'embedding_space.png'
        )
    
    return metrics

def generate_paper_figures(run_dir: Path):
    """Generate publication-ready figures comparing all models."""
    paper_figures_dir = run_dir / 'paper_figures'
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results from all models
    models = ['cnn_baseline', 'lstm_baseline', 'transformer_baseline', 'voice_graph_net']
    datasets = ['kcl', 'italian']
    
    results = {
        'independent': {model: {} for model in models},
        'cross_dataset': {model: {} for model in models}
    }
    
    # Collect independent results
    for model in models:
        for dataset in datasets:
            model_dir = run_dir / model / dataset
            if not model_dir.exists():
                continue
                
            results_file = model_dir / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results['independent'][model][dataset] = json.load(f)
    
    # Collect cross-dataset results
    for model in models:
        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset == test_dataset:
                    continue
                    
                results_dir = run_dir / 'cross_dataset_analysis' / model / f'{train_dataset}_to_{test_dataset}'
                if not results_dir.exists():
                    continue
                    
                results_file = results_dir / 'results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results['cross_dataset'][model][f'{train_dataset}_to_{test_dataset}'] = json.load(f)
    
    # Generate plots
    
    # 1. Box plot comparing AUC across models and datasets
    plt.figure(figsize=(12, 6))
    
    # Independent analysis
    independent_aucs = {model: [] for model in models}
    for model in models:
        for dataset in datasets:
            if dataset in results['independent'][model]:
                independent_aucs[model].append(results['independent'][model][dataset]['metrics']['auc'])
    
    plt.subplot(1, 2, 1)
    plt.boxplot([independent_aucs[model] for model in models], tick_labels=models)
    plt.title('Independent Analysis AUC')
    plt.xticks(rotation=45)
    plt.ylabel('AUC')
    
    # Cross-dataset analysis
    cross_dataset_aucs = {model: [] for model in models}
    for model in models:
        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset == test_dataset:
                    continue
                key = f'{train_dataset}_to_{test_dataset}'
                if key in results['cross_dataset'][model]:
                    cross_dataset_aucs[model].append(results['cross_dataset'][model][key]['metrics']['auc'])
    
    plt.subplot(1, 2, 2)
    plt.boxplot([cross_dataset_aucs[model] for model in models], tick_labels=models)
    plt.title('Cross-Dataset Analysis AUC')
    plt.xticks(rotation=45)
    plt.ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig(paper_figures_dir / 'model_comparison_auc.png')
    plt.close()
    
    # 2. ROC curves for each analysis type
    plt.figure(figsize=(15, 10))
    
    for i, model in enumerate(models, 1):
        plt.subplot(2, 2, i)
        
        # Plot independent ROC curves
        for dataset in datasets:
            if dataset in results['independent'][model]:
                predictions_file = run_dir / model / dataset / 'predictions.json'
                if predictions_file.exists():
                    with open(predictions_file, 'r') as f:
                        preds = json.load(f)
                        if 'labels' in preds:
                            labels = preds['labels']
                            probs = preds['probabilities']
                        else:
                            labels = preds['true_labels']
                            probs = preds['predicted_probabilities']
                        fpr, tpr, _ = roc_curve(labels, probs)
                        plt.plot(fpr, tpr, label=f'{dataset} (AUC={results["independent"][model][dataset]["metrics"]["auc"]:.3f})')
        
        # Plot cross-dataset ROC curves
        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset == test_dataset:
                    continue
                key = f'{train_dataset}_to_{test_dataset}'
                if key in results['cross_dataset'][model]:
                    predictions_file = run_dir / 'cross_dataset_analysis' / model / key / 'predictions.json'
                    if predictions_file.exists():
                        with open(predictions_file, 'r') as f:
                            preds = json.load(f)
                            if 'labels' in preds:
                                labels = preds['labels']
                                probs = preds['probabilities']
                            else:
                                labels = preds['true_labels']
                                probs = preds['predicted_probabilities']
                            fpr, tpr, _ = roc_curve(labels, probs)
                            plt.plot(fpr, tpr, '--', label=f'{key} (AUC={results["cross_dataset"][model][key]["metrics"]["auc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model} ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(paper_figures_dir / 'roc_curves.png', bbox_inches='tight')
    plt.close()
    
    # 3. Confidence intervals for accuracy and AUC
    metrics = ['accuracy', 'auc']
    analysis_types = ['independent', 'cross_dataset']
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        for i, analysis_type in enumerate(analysis_types, 1):
            plt.subplot(1, 2, i)
            
            values = {model: [] for model in models}
            for model in models:
                if analysis_type == 'independent':
                    for dataset in datasets:
                        if dataset in results[analysis_type][model]:
                            values[model].append(results[analysis_type][model][dataset]['metrics'][metric])
                else:
                    for train_dataset in datasets:
                        for test_dataset in datasets:
                            if train_dataset == test_dataset:
                                continue
                            key = f'{train_dataset}_to_{test_dataset}'
                            if key in results[analysis_type][model]:
                                values[model].append(results[analysis_type][model][key]['metrics'][metric])
            
            means = []
            cis = []
            for model in models:
                if len(values[model]) >= 2:
                    mean = np.mean(values[model])
                    std_err = stats.sem(values[model])
                    ci = stats.t.interval(confidence=0.95, df=len(values[model])-1, loc=mean, scale=std_err)
                    means.append(mean)
                    cis.append([ci[0], ci[1]])
                else:
                    means.append(np.nan)
                    cis.append([np.nan, np.nan])
            
            cis = np.array(cis)
            plt.errorbar(range(len(models)), means, yerr=(means-cis[:,0], cis[:,1]-means), fmt='o', capsize=5)
            plt.xticks(range(len(models)), models, rotation=45)
            plt.title(f'{analysis_type} {metric.upper()} with 95% CI')
            plt.ylabel(metric.upper())
            
        plt.tight_layout()
        plt.savefig(paper_figures_dir / f'{metric}_confidence_intervals.png')
        plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn_baseline')
    parser.add_argument('--dataset', type=str, default='kcl')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ultra-debug', action='store_true')
    parser.add_argument('--save-dir', type=str, default='results')
    parser.add_argument('--k-temporal', type=int, default=5)
    parser.add_argument('--k-similarity', type=int, default=5)
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = 'debug' if args.debug else 'full'
    run_dir = Path(args.save_dir) / f'{mode}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run metadata
    with open(run_dir / 'run_metadata.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Determine which models and datasets to run
    models_to_run = ['cnn_baseline', 'lstm_baseline', 'transformer_baseline', 'voice_graph_net'] if args.model == 'all' else [args.model]
    datasets_to_run = ['kcl', 'italian'] if args.dataset == 'all' else [args.dataset]
    
    # Train and evaluate models
    for model_name in models_to_run:
        for dataset_name in datasets_to_run:
            print(f"\n{'='*50}")
            print(f"Training {model_name} on {dataset_name.upper()}")
            print(f"{'='*50}\n")
            
            # Train model
            results, predictions, model = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.workers,
            device=args.device,
                run_dir=run_dir,
            debug=args.debug,
                ultra_debug=args.ultra_debug,
            k_temporal=args.k_temporal,
                k_similarity=args.k_similarity,
                epochs=args.epochs
            )
            
            # Cross-dataset evaluation
            for test_dataset in datasets_to_run:
                if test_dataset == dataset_name:
                    continue
                
                print(f"\n{'-'*20}")
                print(f"Cross-dataset evaluation: {dataset_name} -> {test_dataset}")
                print(f"{'-'*20}\n")
                
                checkpoint_path = run_dir / model_name / dataset_name / 'model.pt'
                cross_dataset_validation(
                    model_name=model_name,
                    train_dataset=dataset_name,
                    test_dataset=test_dataset,
                    checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            num_workers=args.workers,
            device=args.device,
                    run_dir=run_dir,
            debug=args.debug,
                    ultra_debug=args.ultra_debug,
            k_temporal=args.k_temporal,
                    k_similarity=args.k_similarity
                )
        
    # Generate paper figures
    print("\nGenerating paper figures...")
    generate_paper_figures(run_dir)
    
    print("\nOverall Metrics:")
    for model_name in models_to_run:
        print(f"\n{model_name}:")
        for dataset in datasets_to_run:
            try:
                with open(run_dir / model_name / dataset / 'results.json', 'r') as f:
                    results = json.load(f)
                    if 'participant_metrics' in results:
                        print(f"\n{dataset.upper()}:")
                        for metric in ['accuracy', 'auc']:
                            if metric in results['participant_metrics']:
                                print_metrics_with_ci(results['participant_metrics'][metric], metric.upper())
            except Exception as e:
                print(f"Error loading results for {model_name} on {dataset}: {str(e)}")

if __name__ == '__main__':
    main() 