import logging
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ANCAcousticSimulator:
    """
    Simulates the physical acoustic environment where the ambient noise
    and the generated anti-noise waveforms destructively interfere (superposition).
    """

    def __init__(self, target_sr: int = 16000, hardware_latency_ms: int = 3):
        self.target_sr = target_sr
        # The physical time it takes for hardware to process and output the sound
        self.latency_samples = int((hardware_latency_ms / 1000.0) * self.target_sr)

    def _calculate_db_reduction(self, original_signal: np.ndarray, residual_signal: np.ndarray) -> float:
        """
        Calculates the volume reduction in Decibels (dB) by comparing the 
        Root Mean Square (RMS) energy of the signals.
        """
        rms_original = np.sqrt(np.mean(original_signal**2) + 1e-10)
        rms_residual = np.sqrt(np.mean(residual_signal**2) + 1e-10)
        
        # Calculate reduction. Positive dB means volume decreased.
        db_reduction = 20 * np.log10(rms_original / rms_residual)
        return db_reduction

    def simulate_air_superposition(self, original_path: Path, anti_noise_path: Path, output_path: Path) -> None:
        """
        Loads both audio files, simulates the physical hardware delay, 
        adds the waveforms together (interference), and calculates the performance.
        """
        logging.info("Starting acoustic superposition simulation...")

        try:
            # 1. Load the files using librosa to enforce target sample rate automatically
            original_audio, _ = librosa.load(original_path, sr=self.target_sr)
            anti_audio, _ = librosa.load(anti_noise_path, sr=self.target_sr)

            # Ensure both arrays are the exact same length for element-wise addition
            min_length = min(len(original_audio), len(anti_audio))
            original_audio = original_audio[:min_length]
            anti_audio = anti_audio[:min_length]

            # 2. Simulate Physical Reality (Hardware Delay)
            # The anti-noise was predicted (shifted left) in software. 
            # In the real world, the speaker takes time to play it, effectively shifting it back right.
            speaker_output_in_air = np.roll(anti_audio, self.latency_samples)
            
            # The first few samples are empty because the hardware was "thinking"
            speaker_output_in_air[:self.latency_samples] = 0.0

            # 3. Wave Superposition (The Cancellation)
            # Air adds the ambient noise and the speaker output together
            residual_noise = original_audio + speaker_output_in_air

            # 4. Metrics & Export
            db_reduction = self._calculate_db_reduction(original_audio, residual_noise)
            
            # sf.write works fine here because we are writing the 16kHz array to disk
            sf.write(output_path, residual_noise, self.target_sr)
            
            print(f"\n{'='*55}")
            print(f"🔬 ANC VERIFICATION RESULTS")
            print(f"{'='*55}")
            print(f"Original Audio    : {original_path.name}")
            print(f"Anti-Noise Audio  : {anti_noise_path.name}")
            print(f"Residual Audio    : {output_path.name}")
            print(f"{'-'*55}")
            print(f"📉 Noise Reduction: {db_reduction:.2f} dB")
            print(f"{'='*55}\n")

        except Exception as e:
            logging.error(f"Simulation failed: {e}")

# --- Execution Entry Point ---
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    TEST_DIR = BASE_DIR / "data" / "test_samples"
    
    ORIGINAL_FILE = TEST_DIR / "test_sound.wav"
    ANTI_NOISE_FILE = TEST_DIR / "anti_phase_test_sound.wav" # Ensure this matches your previous output
    RESIDUAL_FILE = TEST_DIR / "residual_silence.wav"
    
    if ORIGINAL_FILE.exists() and ANTI_NOISE_FILE.exists():
        simulator = ANCAcousticSimulator(target_sr=16000, hardware_latency_ms=3)
        simulator.simulate_air_superposition(
            original_path=ORIGINAL_FILE, 
            anti_noise_path=ANTI_NOISE_FILE, 
            output_path=RESIDUAL_FILE
        )
    else:
        logging.error("Input files missing. Run 'simulate_anc.py' first to generate anti-noise.")