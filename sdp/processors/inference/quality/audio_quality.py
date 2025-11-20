# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Audio quality assessment processors for TTS data filtering.
"""

import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, BaseProcessor
from sdp.utils.common import load_manifest, save_manifest


class SquimQualityMetrics(BaseParallelProcessor):
    """Calculate audio quality metrics using SQUIM (Speech Quality and Intelligibility Measures).
    
    This processor calculates three key quality metrics for TTS data:
    
    - **PESQ** (Perceptual Evaluation of Speech Quality): 1.0-5.0 scale
      Measures overall speech quality. Higher is better.
      Good TTS: > 3.0
      
    - **STOI** (Short-Time Objective Intelligibility): 0.0-1.0 scale
      Measures speech intelligibility. Higher is better.
      Good TTS: > 0.8
      
    - **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio): dB scale
      Measures signal-to-distortion ratio. Higher is better.
      Good TTS: > 15 dB (clean speech typically 15-20 dB)
    
    Args:
        audio_filepath_key (str): Key for audio file path in manifest. Defaults to "audio_filepath"
        output_pesq_key (str): Output key for PESQ score. Defaults to "pesq"
        output_stoi_key (str): Output key for STOI score. Defaults to "stoi"
        output_sisdr_key (str): Output key for SI-SDR score. Defaults to "sisdr"
        device (str): Device to run model on ("cuda" or "cpu"). Defaults to "cuda"
        batch_size (int): Batch size for processing. Defaults to 16
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.SquimQualityMetrics
              output_manifest_file: ${manifest_dir}/with_quality_metrics.json
              audio_filepath_key: audio_filepath
              device: cuda
              batch_size: 32
    """
    
    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        output_pesq_key: str = "pesq",
        output_stoi_key: str = "stoi",
        output_sisdr_key: str = "sisdr",
        device: str = "cuda",
        batch_size: int = 16,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.output_pesq_key = output_pesq_key
        self.output_stoi_key = output_stoi_key
        self.output_sisdr_key = output_sisdr_key
        self.device = device
        self.batch_size = batch_size
        
        # Load SQUIM model
        if not torch.cuda.is_available() and device == "cuda":
            logger.warning("CUDA not available, using CPU")
            self.device = "cpu"
        
        logger.info("Loading SQUIM quality assessment model...")
        if self.device == "cuda":
            self.model = SQUIM_OBJECTIVE.get_model().cuda()
        else:
            self.model = SQUIM_OBJECTIVE.get_model()
        self.model.eval()
        logger.info("SQUIM model loaded successfully")
    
    def process_dataset_entry(self, data_entry):
        """Process a single dataset entry."""
        audio_path = data_entry[self.audio_filepath_key]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            
            # Resample to 16kHz if needed (SQUIM requires 16kHz)
            if sr != 16000:
                audio_tensor = F.resample(audio_tensor, sr, 16000)
            
            # Move to device
            if self.device == "cuda":
                audio_tensor = audio_tensor.cuda()
            
            # Calculate metrics
            with torch.no_grad():
                stoi, pesq, sisdr = self.model(audio_tensor)
            
            # Add to data entry
            data_entry[self.output_stoi_key] = round(stoi.item(), 4)
            data_entry[self.output_pesq_key] = round(pesq.item(), 4)
            data_entry[self.output_sisdr_key] = round(sisdr.item(), 4)
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics for {audio_path}: {e}")
            data_entry[self.output_stoi_key] = 0.0
            data_entry[self.output_pesq_key] = 0.0
            data_entry[self.output_sisdr_key] = 0.0
        
        return [data_entry]


class BandwidthEstimation(BaseParallelProcessor):
    """Estimate audio bandwidth by analyzing the frequency spectrum.
    
    Bandwidth is important for TTS because:
    - Low bandwidth (<4kHz) = telephone quality, not suitable
    - Medium bandwidth (4-8kHz) = narrowband, acceptable
    - High bandwidth (>8kHz) = wideband, ideal for TTS
    
    Args:
        audio_filepath_key (str): Key for audio file path. Defaults to "audio_filepath"
        output_bandwidth_key (str): Output key for bandwidth in Hz. Defaults to "bandwidth_hz"
        n_fft (int): FFT size. Defaults to 2048
        frequency_threshold_db (float): dB threshold below peak. Defaults to -50.0
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.BandwidthEstimation
              output_manifest_file: ${manifest_dir}/with_bandwidth.json
              audio_filepath_key: audio_filepath
    """
    
    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        output_bandwidth_key: str = "bandwidth_hz",
        n_fft: int = 2048,
        frequency_threshold_db: float = -50.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.output_bandwidth_key = output_bandwidth_key
        self.n_fft = n_fft
        self.frequency_threshold_db = frequency_threshold_db
    
    def _estimate_bandwidth(self, audio, sample_rate):
        """Estimate bandwidth by finding highest frequency with significant energy."""
        # Calculate power spectrum
        hop_length = self.n_fft // 4
        spec = librosa.stft(y=audio, n_fft=self.n_fft, hop_length=hop_length, window="blackmanharris")
        power_spec = np.abs(spec) ** 2
        power_spec = np.mean(power_spec, axis=1)  # Average across time
        power_spec_db = librosa.power_to_db(power_spec, ref=np.max)
        
        # Find highest frequency bin above threshold
        peak_db = np.max(power_spec_db)
        threshold = peak_db + self.frequency_threshold_db
        
        bandwidth_hz = 0
        freq_resolution = sample_rate / self.n_fft
        
        for idx in range(len(power_spec_db) - 1, -1, -1):
            if power_spec_db[idx] > threshold:
                bandwidth_hz = idx * freq_resolution
                break
        
        return int(bandwidth_hz)
    
    def process_dataset_entry(self, data_entry):
        """Process a single dataset entry."""
        audio_path = data_entry[self.audio_filepath_key]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Estimate bandwidth
            bandwidth = self._estimate_bandwidth(audio, sr)
            data_entry[self.output_bandwidth_key] = bandwidth
            
        except Exception as e:
            logger.warning(f"Failed to estimate bandwidth for {audio_path}: {e}")
            data_entry[self.output_bandwidth_key] = 0
        
        return [data_entry]


class SNREstimation(BaseParallelProcessor):
    """Estimate Signal-to-Noise Ratio (SNR) of audio.
    
    SNR measures how much louder the speech is compared to background noise.
    Higher SNR = cleaner audio.
    
    - SNR > 20 dB: Very clean (ideal for TTS)
    - SNR 15-20 dB: Clean (good for TTS)
    - SNR 10-15 dB: Acceptable (moderate quality)
    - SNR < 10 dB: Noisy (not recommended for TTS)
    
    Args:
        audio_filepath_key (str): Key for audio file path. Defaults to "audio_filepath"
        output_snr_key (str): Output key for SNR in dB. Defaults to "snr_db"
        frame_length (int): Frame length for analysis. Defaults to 2048
        top_db (int): Threshold for silence detection. Defaults to 30
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.SNREstimation
              output_manifest_file: ${manifest_dir}/with_snr.json
              audio_filepath_key: audio_filepath
    """
    
    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        output_snr_key: str = "snr_db",
        frame_length: int = 2048,
        top_db: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.output_snr_key = output_snr_key
        self.frame_length = frame_length
        self.top_db = top_db
    
    def _estimate_snr(self, audio):
        """Estimate SNR by comparing speech energy to noise energy."""
        # Calculate frame-wise RMS energy
        hop_length = self.frame_length // 4
        frames = librosa.util.frame(audio, frame_length=self.frame_length, hop_length=hop_length)
        rms = np.sqrt(np.mean(frames**2, axis=0))
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Identify speech vs noise frames using threshold
        threshold = np.max(rms_db) - self.top_db
        speech_frames = rms_db > threshold
        noise_frames = ~speech_frames
        
        if not np.any(speech_frames) or not np.any(noise_frames):
            # Can't estimate SNR reliably
            return 0.0
        
        # Calculate average energy for speech and noise
        speech_energy = np.mean(rms[speech_frames]**2)
        noise_energy = np.mean(rms[noise_frames]**2)
        
        if noise_energy == 0:
            return 100.0  # Very clean signal
        
        snr = 10 * np.log10(speech_energy / noise_energy)
        return float(snr)
    
    def process_dataset_entry(self, data_entry):
        """Process a single dataset entry."""
        audio_path = data_entry[self.audio_filepath_key]
        
        try:
            # Load audio
            audio, _ = librosa.load(audio_path, sr=None, mono=True)
            
            # Estimate SNR
            snr = self._estimate_snr(audio)
            data_entry[self.output_snr_key] = round(snr, 2)
            
        except Exception as e:
            logger.warning(f"Failed to estimate SNR for {audio_path}: {e}")
            data_entry[self.output_snr_key] = 0.0
        
        return [data_entry]

