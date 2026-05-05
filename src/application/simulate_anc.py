import logging
import librosa
from pathlib import Path
from typing import List

# Import our custom modular components
from inference import ANCPredictor
from canceller import ActiveNoiseCanceller

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class SystemOrchestrator:
    """
    Coordinates the interaction between the Deep Learning predictor and the 
    physical phase-inversion canceller based on user-defined constraints.
    """
    
    def __init__(self, model_path: Path, categories: List[str], blocked_classes: List[str]):
        self.predictor = ANCPredictor(model_path=model_path, class_labels=categories)
        self.canceller = ActiveNoiseCanceller(target_sr=16000, hardware_latency_ms=3)
        self.blocked_classes = blocked_classes
        
        logging.info(f"System Initialized. Active Filters: {self.blocked_classes}")

    def process_audio_stream(self, input_audio_path: Path, output_dir: Path) -> None:
        """
        Simulates the end-to-end pipeline: Ingestion -> Classification -> Decision -> Cancellation.
        """
        if not input_audio_path.exists():
            logging.error(f"Input file not found: {input_audio_path}")
            return

        print(f"\n{'='*55}")
        print(f"🎙️  PROCESSING AUDIO STREAM: {input_audio_path.name}")
        print(f"{'='*55}")

        # Step 1: Deep Learning Inference
        predictions = self.predictor.predict(input_audio_path)
        if not predictions:
            logging.error("Inference failed. Aborting pipeline.")
            return
            
        top_class = list(predictions.keys())[0]
        confidence = predictions[top_class] * 100

        print(f"-> Detected Event : [{top_class}] (Confidence: {confidence:.2f}%)")
        print(f"-> User Settings  : {self.blocked_classes}")
        print(f"{'-'*55}")

        # Step 2: Decision Matrix
        if top_class in self.blocked_classes:
            print(f"🚨 ACTION TRIGGERED: '{top_class}' matches blocklist.")
            
            # Load raw audio for physical manipulation
            raw_audio, _ = librosa.load(input_audio_path, sr=self.canceller.target_sr)
            
            # Step 3: Anti-Phase Generation
            anti_noise_signal = self.canceller.generate_anti_noise(raw_audio)
            
            # Export the result
            output_file = output_dir / f"anti_phase_{input_audio_path.name}"
            self.canceller.save_signal(anti_noise_signal, output_file)
            
            print(f"✅ SUCCESS: Anti-noise waveform generated and stored.")
        else:
            print(f"🟢 PASSTHROUGH: '{top_class}' is safe. No action taken.")
            
        print(f"{'='*55}\n")


# --- Execution Entry Point ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    MODEL_PATH = BASE_DIR / "saved_models" / "base_models" / "best_mobilenetv2_anc.h5"
    TEST_DIR = BASE_DIR / "data" / "test_samples"
    
    # 1. System Configuration
    TARGET_CATEGORIES = ['siren', 'car_horn', 'engine', 'wind', 'rain', 'keyboard_typing', 'crying_baby', 'dog']
    
    # Simulated User App Preferences (The sounds the user wants to cancel)
    USER_BLOCKED_SOUNDS = ['dog', 'siren', 'car_horn'] 
    
    # 2. Initialize Orchestrator
    orchestrator = SystemOrchestrator(
        model_path=MODEL_PATH, 
        categories=TARGET_CATEGORIES, 
        blocked_classes=USER_BLOCKED_SOUNDS
    )
    
    # 3. Run Pipeline on Test File
    test_file = TEST_DIR / "test_sound.wav"
    orchestrator.process_audio_stream(input_audio_path=test_file, output_dir=TEST_DIR)