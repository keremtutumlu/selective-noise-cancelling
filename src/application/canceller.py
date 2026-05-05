import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ActiveNoiseCanceller:
    """
    Handles the generation of anti-phase audio signals for True Active Noise Cancellation.
    Implements latency compensation for predictive ANC hardware simulation.
    """

    def __init__(self, target_sr: int = 16000, hardware_latency_ms: int = 3):
        """
        Initializes the canceller with hardware-specific constraints.
        
        Args:
            target_sr (int): Sample rate of the audio system (default 16kHz).
            hardware_latency_ms (int): Estimated hardware processing delay in milliseconds.
        """
        self.target_sr = target_sr
        # Convert millisecond latency to sample count (e.g., 3ms at 16kHz = 48 samples)
        self.latency_samples = int((hardware_latency_ms / 1000.0) * self.target_sr)

    def generate_anti_noise(self, raw_audio: np.ndarray) -> np.ndarray:
        """
        Creates an anti-phase signal to physically cancel the incoming noise.
        Applies a phase inversion (-1.0) and shifts the signal to compensate for hardware latency.
        
        Args:
            raw_audio (np.ndarray): The original noise waveform.
            
        Returns:
            np.ndarray: The latency-compensated anti-noise waveform.
        """
        logging.info("Generating anti-phase signal for active cancellation...")
        
        # 1. Phase Inversion (180 degrees)
        anti_noise = -1.0 * raw_audio
        
        # 2. Predictive Shift (Latency Compensation)
        # Shift the array to the left (negative roll) to anticipate the delay
        anti_noise = np.roll(anti_noise, -self.latency_samples)
        
        # Fill the trailing edge with silence (zeros) to prevent wrap-around artifacts
        anti_noise[-self.latency_samples:] = 0.0
        
        return anti_noise

    def save_signal(self, audio_data: np.ndarray, output_path: Path) -> bool:
        """
        Exports the generated waveform to a WAV file for physical testing.
        """
        try:
            sf.write(output_path, audio_data, self.target_sr)
            logging.info(f"Anti-noise signal successfully exported to: {output_path.name}")
            return True
        except Exception as e:
            logging.error(f"Failed to export audio signal: {e}")
            return False