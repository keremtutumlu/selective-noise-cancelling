"""
Module: synthetic_data_generator
Description: 
    This module is responsible for generating a robust, synthetic multi-label dataset 
    for Selective Active Noise Cancellation (SNC). It simulates real-world acoustic 
    environments by superimposing (mixing) multiple isolated audio samples (e.g., 
    wind noise + siren) at varying Signal-to-Noise Ratios (SNR). 
    
    Key Engineering Features:
    - Low-Latency Adaptation: Crops audio into 1-second dynamic windows to meet 
      Edge AI (MCU) real-time processing constraints.
    - Data Augmentation: Randomly selects and mixes 1 to 3 distinct audio classes 
      per sample to prevent model overfitting and improve generalization.
    - Feature Extraction: Computes Log-Mel Spectrograms and applies RGB channel 
      emulation (3 channels) to satisfy the input requirements of MobileNetV2.
    - Multi-Hot Encoding: Transforms single-label data into multi-label arrays 
      (e.g., [1, 0, 1, 0]) suitable for Sigmoid-based neural network architectures.
"""

import logging
import random
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Configure global logging standards
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticDatasetBuilder:
    """
    Orchestrates the extraction, mixing, and feature engineering of audio files 
    to build a multi-label training dataset.
    """
    
    def __init__(self, csv_path: Path, audio_dir: Path, target_classes: List[str], target_sr: int = 16000):
        """
        Initializes the builder and validates the environment constraints.
        """
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.target_classes = sorted(target_classes)  # Enforce consistent alphabetical indexing
        self.target_sr = target_sr
        self.window_length = target_sr * 1  # Exactly 1 second of audio (16,000 samples)
        self.num_classes = len(self.target_classes)
        
        # Build the validated file index immediately upon instantiation
        self.class_file_index = self._build_validated_file_index()

    def _build_validated_file_index(self) -> Dict[str, List[Path]]:
        """
        Reads the metadata CSV, filters for target classes, and strictly validates 
        the physical existence of each audio file before indexing it.
        """
        logging.info("Building and validating file index from metadata...")
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at: {self.csv_path}")
        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found at: {self.audio_dir}")

        df = pd.read_csv(self.csv_path)
        filtered_df = df[df['category'].isin(self.target_classes)]
        
        file_index = {cls: [] for cls in self.target_classes}
        missing_files_count = 0
        
        for _, row in filtered_df.iterrows():
            file_path = self.audio_dir / row['filename']
            
            # Robustness: Check if file physically exists on the disk
            if file_path.exists():
                file_index[row['category']].append(file_path)
            else:
                missing_files_count += 1
                logging.debug(f"Missing file skipped: {file_path.name}")
                
        if missing_files_count > 0:
            logging.warning(f"Skipped {missing_files_count} files that were listed in CSV but missing from disk.")
            
        for cls, files in file_index.items():
            if len(files) == 0:
                logging.error(f"CRITICAL: No valid files found for class '{cls}'. Check dataset integrity.")
            else:
                logging.info(f"Class '{cls}' mapped successfully with {len(files)} validated files.")
            
        return file_index

    def _extract_random_window(self, audio: np.ndarray) -> np.ndarray:
        """
        Extracts a random continuous 1-second block from the provided audio array.
        Applies zero-padding if the source audio is shorter than 1 second.
        """
        audio_length = len(audio)
        
        if audio_length > self.window_length:
            max_start_index = audio_length - self.window_length
            start_idx = random.randint(0, max_start_index)
            return audio[start_idx : start_idx + self.window_length]
        
        elif audio_length < self.window_length:
            pad_width = self.window_length - audio_length
            return np.pad(audio, (0, pad_width), mode='constant')
            
        return audio

    def _apply_superposition(self, audio_tracks: List[np.ndarray]) -> np.ndarray:
        """
        Mixes multiple audio arrays via superposition. Simulates varying SNR 
        by assigning random amplitude weights to each track.
        """
        mixed_signal = np.zeros(self.window_length, dtype=np.float32)
        
        for track in audio_tracks:
            # Apply dynamic volume scaling (SNR variance)
            amplitude_weight = random.uniform(0.4, 1.0)
            mixed_signal += (track * amplitude_weight)
            
        # Peak normalization to prevent digital clipping [-1.0, 1.0]
        peak_value = np.max(np.abs(mixed_signal))
        if peak_value > 0:
            mixed_signal = mixed_signal / peak_value
            
        return mixed_signal

    def _compute_model_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Transforms the raw audio waveform into a Z-score normalized, 3-channel 
        Log-Mel Spectrogram tensor optimized for MobileNetV2 ingestion.
        """
        # Generate Mel Spectrogram (25ms window, 10ms hop)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.target_sr, n_fft=400, hop_length=160, n_mels=64
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Standardization (Z-score)
        mean_val = np.mean(log_mel_spec)
        std_val = np.std(log_mel_spec)
        normalized_features = (log_mel_spec - mean_val) / (std_val + 1e-6)
        
        # Reshape to (Mel_Bins, Time_Steps, Channels) -> (64, 101, 3)
        expanded_tensor = np.expand_dims(normalized_features, axis=-1)
        rgb_emulated_tensor = np.repeat(expanded_tensor, 3, axis=-1)
        
        return rgb_emulated_tensor

    def execute_pipeline(self, num_samples: int, output_dir: Path) -> None:
        """
        Executes the dataset generation loop, compiling the feature matrix (X) 
        and the multi-hot label matrix (y), and serializes them to disk.
        """
        logging.info(f"Initiating generation pipeline for {num_samples} samples...")
        
        feature_matrix = []
        label_matrix = []
        
        for iteration in range(num_samples):
            # 1. Determine mix complexity
            num_sounds_to_mix = random.randint(1, 3)
            selected_classes = random.sample(self.target_classes, num_sounds_to_mix)
            
            # 2. Initialize the multi-hot label vector
            multi_hot_vector = np.zeros(self.num_classes, dtype=np.float32)
            active_audio_crops = []
            
            # 3. Process each selected sound
            for cls in selected_classes:
                class_index = self.target_classes.index(cls)
                multi_hot_vector[class_index] = 1.0  # Activate class flag
                
                # Retrieve and read a random validated file
                file_path = random.choice(self.class_file_index[cls])
                raw_audio, _ = librosa.load(file_path, sr=self.target_sr)
                
                # Extract 1-second window
                cropped_audio = self._extract_random_window(raw_audio)
                active_audio_crops.append(cropped_audio)
                
            # 4. Superposition and Feature Extraction
            mixed_audio_signal = self._apply_superposition(active_audio_crops)
            final_features = self._compute_model_features(mixed_audio_signal)
            
            # 5. Append to dataset matrices
            feature_matrix.append(final_features)
            label_matrix.append(multi_hot_vector)
            
            # Progress tracking
            if (iteration + 1) % 500 == 0:
                logging.info(f"Pipeline status: {iteration + 1} / {num_samples} records compiled.")

        # Serialize datasets
        X_tensor = np.array(feature_matrix, dtype=np.float32)
        y_tensor = np.array(label_matrix, dtype=np.float32)
        
        logging.info(f"Pipeline completed successfully.")
        logging.info(f"Feature Tensor (X) Shape: {X_tensor.shape}")
        logging.info(f"Label Tensor (y) Shape: {y_tensor.shape}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "X_multi_features.npy", X_tensor)
        np.save(output_dir / "y_multi_labels.npy", y_tensor)
        logging.info(f"Binary artifacts exported to: {output_dir}")

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Path configuration
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CSV_PATH = BASE_DIR / "data" / "raw" / "archive" / "esc50.csv"
    AUDIO_DIR = BASE_DIR / "data" / "raw" / "archive" / "audio" / "audio"
    PROCESSED_DIR = BASE_DIR / "data" / "processed" / "training_pipeline"
    
    # Target acoustic environment parameters
    TARGET_CATEGORIES = ['siren', 'car_horn', 'engine', 'wind', 'rain', 'keyboard_typing', 'crying_baby', 'dog']
    
    try:
        # Initialize the builder
        dataset_builder = SyntheticDatasetBuilder(
            csv_path=CSV_PATH, 
            audio_dir=AUDIO_DIR, 
            target_classes=TARGET_CATEGORIES
        )
        
        # Execute the pipeline
        dataset_builder.execute_pipeline(num_samples=5000, output_dir=PROCESSED_DIR)
        
    except Exception as e:
        logging.critical(f"Pipeline execution failed: {e}")