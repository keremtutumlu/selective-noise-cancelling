import logging
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Dict

# Configure logging for real-time inference tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ANCPredictor:
    """
    Handles loading the trained MobileNetV2 model and performing real-time 
    inference on single audio files for the Selective ANC system.
    """
    def __init__(self, model_path: Path, class_labels: List[str], target_sr: int = 16000):
        self.target_sr = target_sr
        
        # LabelEncoder sorts classes alphabetically during training.
        # We MUST sort them here to ensure the output indices match the trained weights.
        self.class_labels = sorted(class_labels) 
        
        logging.info(f"Loading ANC model from {model_path.name}...")
        if not model_path.exists():
            raise FileNotFoundError(f"Model weight file not found at {model_path}")
            
        self.model = tf.keras.models.load_model(model_path)
        logging.info("Model loaded successfully into memory.")

    def preprocess_live_audio(self, file_path: Path) -> np.ndarray:
        """
        Applies the exact same preprocessing pipeline used during dataset generation.
        Transforms raw audio into a 3-channel Log-Mel Spectrogram tensor.
        """
        # 1. Load and resample to target sample rate (16kHz)
        audio, sr = librosa.load(file_path, sr=self.target_sr)
        
        # --- YENİ EKLENEN KISIM: Sesi 5 Saniyeye Sabitleme ---
        # 5 saniye * 16000 Hz = 80000 sample
        target_length = 5 * self.target_sr 
        
        if len(audio) > target_length:
            # Ses 5 saniyeden uzunsa, ilk 5 saniyesini kes al
            audio = audio[:target_length]
        else:
            # Ses 5 saniyeden kısaysa, eksik kalan kısmı sıfırlarla (sessizlik) doldur
            audio = np.pad(audio, (0, max(0, target_length - len(audio))), "constant")
        # -----------------------------------------------------

        # 2. Extract Mel Spectrogram (25ms window, 10ms hop, 64 mels)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_fft=400, 
            hop_length=160, 
            n_mels=64
        )
        
        # 3. Convert to Log scale (Decibels)
        log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 4. Standardize (Z-score normalization)
        mean = np.mean(log_mel)
        std = np.std(log_mel)
        normalized_mel = (log_mel - mean) / (std + 1e-6)
        
        # 5. Reshape for MobileNetV2 Input: (Batch_Size, Mel_Bins, Time_Steps, Channels)
        mel_expanded = np.expand_dims(normalized_mel, axis=(0, -1))
        
        # 6. Duplicate the single grayscale channel 3 times (RGB emulation)
        mel_rgb = np.repeat(mel_expanded, 3, axis=-1)
        
        return mel_rgb

    def predict(self, file_path: Path) -> Dict[str, float]:
        """
        Runs the neural network inference on a given audio file and returns 
        the probability distribution across all targeted classes.
        """
        logging.info(f"Analyzing audio profile for: {file_path.name}")
        
        try:
            features = self.preprocess_live_audio(file_path)
            
            # Perform prediction (verbose=0 suppresses the progress bar for clean console output)
            predictions = self.model.predict(features, verbose=0)[0] 
            
            # Map probabilities to their corresponding class labels
            results = {label: float(prob) for label, prob in zip(self.class_labels, predictions)}
            
            # Sort the dictionary by probability in descending order
            sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            return sorted_results
            
        except Exception as e:
            logging.error(f"Inference pipeline failed for {file_path.name}: {e}")
            return {}

# --- Execution ---
if __name__ == "__main__":
    # Path definitions
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    MODEL_PATH = BASE_DIR / "saved_models" / "base_models" / "best_mobilenetv2_anc.h5"
    
    # The exact categories used during the training phase
    TARGET_CATEGORIES = ['siren', 'car_horn', 'engine', 'wind', 'rain', 'keyboard_typing', 'crying_baby', 'dog']
    
    # Create a directory for raw test samples if it doesn't exist
    TEST_AUDIO_DIR = BASE_DIR / "data" / "test_samples"
    TEST_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # The file we want to test (Ensure this file is placed in the directory before running)
    SAMPLE_AUDIO_FILE = TEST_AUDIO_DIR / "test_sound.wav"
    
    try:
        predictor = ANCPredictor(model_path=MODEL_PATH, class_labels=TARGET_CATEGORIES)
        
        if SAMPLE_AUDIO_FILE.exists():
            results = predictor.predict(SAMPLE_AUDIO_FILE)
            
            print(f"\n{'='*50}")
            print(f"🎯 INFERENCE RESULTS: {SAMPLE_AUDIO_FILE.name}")
            print(f"{'='*50}")
            
            for label, prob in results.items():
                # Highlight the top prediction
                marker = ">> " if prob == max(results.values()) else "   "
                print(f"{marker}{label.ljust(18)}: {prob * 100:>6.2f}%")
                
            print(f"{'='*50}\n")
        else:
            logging.warning(f"No test audio found. Please place a .wav file at:\n{SAMPLE_AUDIO_FILE}")
            
    except Exception as e:
        logging.error(f"Execution critically failed: {e}")