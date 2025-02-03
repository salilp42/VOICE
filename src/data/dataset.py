import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

class VoiceDataset(Dataset):
    def __init__(
        self,
        segments: np.ndarray,
        metadata: pd.DataFrame,
        segment_length: int = 16000,
        normalize: bool = True
    ):
        self.segments = torch.FloatTensor(segments)
        self.metadata = metadata
        self.segment_length = segment_length
        self.normalize = normalize
        
        if normalize:
            # Normalize each segment independently
            print("Normalizing segments...")
            self.segments = F.normalize(self.segments, p=2, dim=1)
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        segment = self.segments[idx]
        label = torch.tensor(self.metadata.iloc[idx]['clean_label'], dtype=torch.long)
        participant_id = self.metadata.iloc[idx]['participant_id']
        
        return {
            'segment': segment,
            'label': label,
            'participant_id': participant_id
        }

class VoiceDataModule:
    def __init__(
        self,
        dataset_name: str,
        segment_length: int = 16000,
        batch_size: int = 32,
        num_workers: int = 4,
        n_splits: int = 5,
        normalize: bool = True,
        debug: bool = False,
        ultra_debug: bool = False  # New parameter for ultra-debug mode
    ):
        self.dataset_name = dataset_name
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_splits = n_splits
        self.normalize = normalize
        self.debug = debug
        self.ultra_debug = ultra_debug
        
        self.load_data()
        
    def load_data(self):
        """Load segments and metadata based on dataset name."""
        print(f"\nLoading {self.dataset_name.upper()} dataset...")
        
        if self.dataset_name.lower() == 'kcl':
            self.segments = np.load("Processed_Data_Complete/raw_segments/KCL_segments.npy")
            self.metadata = pd.read_csv("Processed_Data_Complete/KCL_metadata_clean.csv")
        elif self.dataset_name.lower() == 'italian':
            self.segments = np.load("Processed_Data_Complete/raw_segments/Italian_segments.npy")
            self.metadata = pd.read_csv("Processed_Data_Complete/Italian_metadata_clean.csv")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        if self.debug or self.ultra_debug:
            print("Debug mode: Selecting subset of participants...")
            # Get participants from each class
            control_participants = self.metadata[self.metadata['clean_label'] == 0]['participant_id'].unique()
            pd_participants = self.metadata[self.metadata['clean_label'] == 1]['participant_id'].unique()
            
            np.random.seed(42)  # For reproducibility
            
            if self.ultra_debug:
                # Ultra-debug: Use exactly 10 participants per class
                n_controls = n_pd = 12
                print("Ultra-debug mode: Using 10 participants per class")
            else:
                # Regular debug: Use up to 50 participants per class
                n_controls = min(50, len(control_participants))
                n_pd = min(50, len(pd_participants))
                print(f"Debug mode: Using up to 50 participants per class")
            
            # Stratified selection of participants
            selected_controls = np.random.choice(control_participants, size=n_controls, replace=False)
            selected_pd = np.random.choice(pd_participants, size=n_pd, replace=False)
            selected_participants = np.concatenate([selected_controls, selected_pd])
            
            # Filter data
            mask = self.metadata['participant_id'].isin(selected_participants)
            self.metadata = self.metadata[mask].reset_index(drop=True)
            self.segments = self.segments[mask]
            
            print(f"Selected {len(selected_participants)} participants ({len(selected_controls)} control, {len(selected_pd)} PD)")
            
            if self.ultra_debug:
                # In ultra-debug mode, limit segments per participant for faster iteration
                segments_per_participant = 100
                participant_segments = defaultdict(int)
                keep_mask = []
                
                for idx, participant in enumerate(self.metadata['participant_id']):
                    if participant_segments[participant] < segments_per_participant:
                        keep_mask.append(True)
                        participant_segments[participant] += 1
                    else:
                        keep_mask.append(False)
                
                # Apply segment limit
                keep_mask = np.array(keep_mask)
                self.metadata = self.metadata[keep_mask].reset_index(drop=True)
                self.segments = self.segments[keep_mask]
                print(f"Limited to {segments_per_participant} segments per participant in ultra-debug mode")
        
        print(f"Final dataset: {len(self.segments)} segments from {len(self.metadata['participant_id'].unique())} participants")
        print(f"Label distribution: {self.metadata['clean_label'].value_counts().to_dict()}")
    
    def get_participant_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate participant-level stratified splits with overlap checking."""
        print("\nGenerating participant-level stratified splits...")
        # Get unique participants and their labels
        participant_labels = self.metadata.groupby('participant_id')['clean_label'].first()
        
        if self.ultra_debug:
            print("Ultra-debug mode: Using simple train/val split...")
            # Get participants for each class
            control_participants = participant_labels[participant_labels == 0].index
            pd_participants = participant_labels[participant_labels == 1].index
            
            # Split participants with more for training (8 train, 4 val per class)
            train_control = control_participants[:8]
            train_pd = pd_participants[:8]
            val_control = control_participants[8:]
            val_pd = pd_participants[8:]
            
            # Combine participants
            train_participants = np.concatenate([train_control, train_pd])
            val_participants = np.concatenate([val_control, val_pd])
            
            # Verify no overlap
            assert len(set(train_participants) & set(val_participants)) == 0, "Participant overlap detected!"
            
            # Get segment indices for each split
            train_mask = self.metadata['participant_id'].isin(train_participants)
            val_mask = self.metadata['participant_id'].isin(val_participants)
            
            return [(np.where(train_mask)[0], np.where(val_mask)[0])]
        else:
            # Regular k-fold cross-validation with overlap checking
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            participant_ids = participant_labels.index
            participant_labels_array = participant_labels.values
            
            splits = []
            for train_idx, val_idx in skf.split(participant_ids, participant_labels_array):
                train_participants = participant_ids[train_idx]
                val_participants = participant_ids[val_idx]
                
                # Verify no overlap
                assert len(set(train_participants) & set(val_participants)) == 0, "Participant overlap detected!"
                
                # Get segment indices for each split
                train_mask = self.metadata['participant_id'].isin(train_participants)
                val_mask = self.metadata['participant_id'].isin(val_participants)
                
                splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
            
            return splits
    
    def get_fold_dataloaders(
        self,
        fold_idx: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders for a specific fold."""
        splits = self.get_participant_splits()
        train_idx, val_idx = splits[fold_idx]
        
        print(f"\nPreparing fold {fold_idx + 1}/{self.n_splits}")
        print(f"Train: {len(train_idx)} segments")
        print(f"Val: {len(val_idx)} segments")
        
        # Create train dataset
        train_dataset = VoiceDataset(
            segments=self.segments[train_idx],
            metadata=self.metadata.iloc[train_idx],
            segment_length=self.segment_length,
            normalize=self.normalize
        )
        
        # Create validation dataset
        val_dataset = VoiceDataset(
            segments=self.segments[val_idx],
            metadata=self.metadata.iloc[val_idx],
            segment_length=self.segment_length,
            normalize=self.normalize
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_participant_dataloader(
        self,
        participant_id: str,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """Get a dataloader for a specific participant's segments."""
        mask = self.metadata['participant_id'] == participant_id
        participant_segments = self.segments[mask]
        participant_metadata = self.metadata[mask]
        
        dataset = VoiceDataset(
            segments=participant_segments,
            metadata=participant_metadata,
            segment_length=self.segment_length,
            normalize=self.normalize
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 