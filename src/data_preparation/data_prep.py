import logging
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

# Configure logging for tracking the process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioPreprocessor:
    """
    A class to handle audio loading, feature extraction (Log-Mel Spectrogram), 
    and dataset formatting for Deep Learning models.
    """
    def __init__(self, target_sr: int = 16000, n_mels: int = 64, n_fft: int = 400, hop_length: int = 160):
        """
        Initializes the preprocessor with specific audio parameters.
        
        Args:
            target_sr (int): Target sample rate (16kHz).
            n_mels (int): Number of Mel bands to generate.
            n_fft (int): Length of the FFT window (25ms window = 400 samples at 16kHz).
            hop_length (int): Number of samples between successive frames (10ms = 160 samples).
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, file_path: Path) -> Optional[np.ndarray]:
        """
        Loads an audio file and computes its Log-Mel Spectrogram.
        
        Args:
            file_path (Path): The path to the audio file.
            
        Returns:
            np.ndarray: Normalized Log-Mel Spectrogram, or None if an error occurs.
        """
        try:
            # 1. Load and resample audio
            audio, sr = librosa.load(file_path, sr=self.target_sr)
            
            # 2. Extract Mel Spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                n_mels=self.n_mels
            )
            
            # 3. Convert to Log scale (Decibels)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # 4. Standardize (Z-score normalization)
            mean = np.mean(log_mel_spectrogram)
            std = np.std(log_mel_spectrogram)
            normalized_features = (log_mel_spectrogram - mean) / (std + 1e-6)
            
            return normalized_features
            
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return None

    def process_esc50_dataset(self, csv_path: Path, audio_dir: Path, target_classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses the ESC-50 metadata, filters target classes, and extracts features for the entire dataset.
        
        Args:
            csv_path (Path): Path to the ESC-50 metadata CSV.
            audio_dir (Path): Directory containing the raw audio files.
            target_classes (List[str]): List of class names to keep.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Extracted features and corresponding labels.
        """
        logging.info("Starting ESC-50 dataset processing...")
        
        if not csv_path.exists() or not audio_dir.exists():
            logging.error("Dataset paths are invalid. Please check your directories.")
            return np.array([]), np.array([])

        df = pd.read_csv(csv_path)
        filtered_df = df[df['category'].isin(target_classes)]
        
        features_list = []
        labels_list = []
        
        total_files = len(filtered_df)
        logging.info(f"Found {total_files} files matching target classes.")
        
        for index, row in filtered_df.iterrows():
            file_name = row['filename']
            label = row['category']
            file_path = audio_dir / file_name
            
            features = self.extract_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
                
            if (index + 1) % 50 == 0:
                logging.info(f"Processed {index + 1} / {total_files} files...")
                
        logging.info("Dataset processing complete.")
        return np.array(features_list), np.array(labels_list)

# --- Usage Example ---
if __name__ == "__main__":
    # Define paths using pathlib for OS independence
    BASE_DIR = Path(__file__).parent.parent.parent
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    
    CSV_FILE = RAW_DATA_DIR / "archive" / "esc50.csv"
    AUDIO_FOLDER = RAW_DATA_DIR / "archive"/ "audio" / "audio" / "16000"
    
    # Define which classes we want to classify for the ANC project
    TARGET_CATEGORIES = ['siren', 'car_horn', 'engine', 'wind', 'rain', 'keyboard_typing', 'crying_baby', 'dog']
    
    # Ensure processed directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Preprocessor (25ms window, 10ms hop length, 64 Mel filters based on proposal)
    preprocessor = AudioPreprocessor(target_sr=16000, n_mels=64, n_fft=400, hop_length=160)
    
    # Run pipeline
    X, y = preprocessor.process_esc50_dataset(CSV_FILE, AUDIO_FOLDER, TARGET_CATEGORIES)
    
    # Save the processed numpy arrays for the training branch
    if len(X) > 0:
        np.save(PROCESSED_DATA_DIR / "X_features.npy", X)
        np.save(PROCESSED_DATA_DIR / "y_labels.npy", y)
        logging.info("Features and labels successfully saved to disk.")