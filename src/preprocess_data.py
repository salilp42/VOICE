# Preprocessing script for voice-based PD detection
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def detect_voice_activity(
    audio: np.ndarray,
    sr: int = 16000,
    frame_length: int = 1024,
    hop_length: int = 512,
    energy_threshold: float = 0.1,
    zcr_threshold: float = 0.2
) -> np.ndarray:
    """
    Detect voice activity in audio using energy and zero-crossing rate.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        frame_length: Length of each frame
        hop_length: Number of samples between frames
        energy_threshold: Energy threshold for voice activity
        zcr_threshold: Zero-crossing rate threshold
    
    Returns:
        Binary mask indicating voice activity
    """
    # Compute short-time energy
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy = energy / np.max(energy)
    
    # Compute zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = zcr / np.max(zcr)
    
    # Create voice activity mask
    mask = (energy > energy_threshold) & (zcr < zcr_threshold)
    
    # Expand mask to match audio length
    full_mask = np.zeros_like(audio)
    for i, active in enumerate(mask):
        start = i * hop_length
        end = start + frame_length
        if active:
            full_mask[start:end] = 1
            
    return full_mask

def segment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    segment_length: int = 32000,  # 2 seconds
    overlap: float = 0.5,
    min_segments: int = 3
) -> List[np.ndarray]:
    """
    Segment audio into fixed-length segments with overlap.
    
    Args:
        audio: Input audio signal
        sr: Sampling rate
        segment_length: Length of each segment in samples
        overlap: Overlap between segments (0-1)
        min_segments: Minimum number of segments required
    
    Returns:
        List of audio segments
    """
    # Calculate hop length
    hop_length = int(segment_length * (1 - overlap))
    
    # Calculate number of segments
    n_segments = 1 + (len(audio) - segment_length) // hop_length
    
    if n_segments < min_segments:
        return []
    
    # Extract segments
    segments = []
    for i in range(n_segments):
        start = i * hop_length
        end = start + segment_length
        segment = audio[start:end]
        
        # Only keep if segment is complete
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments

def preprocess_audio(
    audio_path: str,
    target_sr: int = 16000,
    segment_length: int = 32000,  # 2 seconds
    overlap: float = 0.5
) -> Tuple[List[np.ndarray], Dict]:
    """
    Preprocess a single audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
        segment_length: Length of each segment in samples
        overlap: Overlap between segments (0-1)
    
    Returns:
        List of preprocessed segments and metadata
    """
    # Load and resample audio
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Remove DC offset
    audio = audio - np.mean(audio)
    
    # Normalize power
    audio = audio / np.sqrt(np.mean(audio**2))
    
    # Detect voice activity
    voice_mask = detect_voice_activity(audio)
    audio = audio * voice_mask
    
    # Remove silence
    audio = audio[audio != 0]
    
    # Segment audio
    segments = segment_audio(audio, sr=target_sr, segment_length=segment_length, overlap=overlap)
    
    # Create metadata
    metadata = {
        'original_duration': len(audio) / sr,
        'n_segments': len(segments),
        'sampling_rate': target_sr,
        'segment_length': segment_length,
        'overlap': overlap
    }
    
    return segments, metadata

def process_dataset(
    data_dir: str,
    dataset: str,
    label_map: Optional[Dict[str, int]] = None
) -> None:
    """
    Process all audio files in a dataset.
    
    Args:
        data_dir: Path to data directory
        dataset: Dataset name ('KCL' or 'Italian')
        label_map: Optional mapping of participant IDs to labels
    """
    data_dir = Path(data_dir)
    save_dir = Path("Processed_Data_Complete")
    save_dir.mkdir(exist_ok=True)
    
    # Create directories
    segments_dir = save_dir / "raw_segments"
    segments_dir.mkdir(exist_ok=True)
    
    # Initialize lists for segments and metadata
    all_segments = []
    metadata_records = []
    
    # Process each audio file
    audio_files = list(data_dir.rglob("*.wav"))
    print(f"\nProcessing {dataset} dataset ({len(audio_files)} files)...")
    
    for audio_path in tqdm(audio_files):
        try:
            # Extract participant ID from path
            participant_id = audio_path.parent.name
            
            # Get label if provided
            label = label_map.get(participant_id, -1) if label_map else -1
            
            # Process audio
            segments, segment_metadata = preprocess_audio(str(audio_path))
            
            if segments:  # Only include if segments were extracted
                # Add segments to list
                all_segments.extend(segments)
                
                # Create metadata record for each segment
                for i in range(len(segments)):
                    metadata_records.append({
                        'participant_id': participant_id,
                        'label': label,
                        'segment_idx': i,
                        'audio_file': audio_path.name,
                        **segment_metadata
                    })
        
        except Exception as e:
            print(f"\nError processing {audio_path}: {str(e)}")
            continue
    
    # Convert segments to numpy array
    all_segments = np.array(all_segments)
    
    # Save segments
    np.save(segments_dir / f"{dataset}_segments.npy", all_segments)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(save_dir / f"{dataset}_metadata.csv", index=False)
    
    print(f"\nSaved {len(all_segments)} segments from {len(metadata_records)} files")
    print(f"Segments shape: {all_segments.shape}")
    print(f"Metadata shape: {metadata_df.shape}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--dataset', type=str, choices=['KCL', 'Italian'], required=True, help='Dataset to process')
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(args.data_dir, args.dataset)

if __name__ == '__main__':
    main() 