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
Source Separation Processor using UVR-MDX-NET model.

Based on Amphion's Emilia preprocessing pipeline.
Separates vocals from background music and noise.
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import onnxruntime as ort
from tqdm import tqdm
from typing import List

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class ConvTDFNet:
    """
    ConvTDFNet - Convolutional Temporal Frequency Domain Network for source separation.
    """

    def __init__(self, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        """
        Initialize ConvTDFNet.

        Args:
            target_name (str): The target name for separation.
            L (int): Number of layers.
            dim_f (int): Dimension in the frequency domain.
            dim_t (int): Dimension in the time domain (log2).
            n_fft (int): FFT size.
            hop (int, optional): Hop size. Defaults to 1024.
        """
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.target_name = target_name

        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c

        self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x):
        """Perform Short-Time Fourier Transform (STFT)."""
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        """Perform Inverse Short-Time Fourier Transform (ISTFT)."""
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])


class Predictor:
    """
    Predictor class for source separation using ConvTDFNet and ONNX Runtime.
    """

    def __init__(self, model_path, device, denoise=True, margin=44100, chunks=15, 
                 n_fft=6144, dim_t=8, dim_f=3072):
        """
        Initialize the Predictor.

        Args:
            model_path (str): Path to the ONNX model file
            device (str): Device to run the model ('cuda' or 'cpu')
            denoise (bool): Whether to apply denoising
            margin (int): Margin for chunking
            chunks (int): Number of chunks
            n_fft (int): FFT size
            dim_t (int): Time dimension
            dim_f (int): Frequency dimension
        """
        self.denoise = denoise
        self.margin = margin
        self.chunks = chunks
        self.n_fft = n_fft
        self.dim_t = dim_t
        self.dim_f = dim_f
        
        self.model_ = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=dim_f,
            dim_t=dim_t,
            n_fft=n_fft,
        )

        if device == "cuda":
            self.model = ort.InferenceSession(
                model_path, providers=["CUDAExecutionProvider"]
            )
        elif device == "cpu":
            self.model = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        else:
            raise ValueError("Device must be either 'cuda' or 'cpu'")

    def demix(self, mix):
        """Separate the sources from the input mix."""
        samples = mix.shape[-1]
        margin = self.margin
        chunk_size = self.chunks * 44100

        assert margin != 0, "Margin cannot be zero!"

        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        """Base function for source separation."""
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes), desc="Source separation", leave=False)

        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size

            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32)

            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.denoise:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0])
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size

                if margin_size == 0:
                    end = None

                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)

        progress_bar.close()
        return _sources

    def predict(self, mix):
        """Predict the separated sources from the input mix."""
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])

        tail = mix.shape[1] % (self.chunks * 44100)
        if mix.shape[1] % (self.chunks * 44100) != 0:
            mix = np.pad(
                mix,
                (
                    (0, 0),
                    (
                        0,
                        self.chunks * 44100
                        - mix.shape[1] % (self.chunks * 44100),
                    ),
                ),
            )

        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T

        if tail != 0:
            return opt[: -(self.chunks * 44100 - tail), :]
        else:
            return opt


class SourceSeparation(BaseParallelProcessor):
    """
    Source separation processor to remove background music and noise from audio.

    Uses the UVR-MDX-NET-Inst_HQ_3 model to separate vocals from background music.
    Replaces the original audio file with the separated vocals.

    Args:
        model_path (str): Path to the UVR-MDX-NET ONNX model file.
            Download from: https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx
        audio_filepath_key (str): Key in manifest for audio file path. Default: "audio_filepath"
        output_dir (str): Directory to save separated audio files. If None, overwrites original.
        device (str): Device to run model on ('cuda' or 'cpu'). Default: 'cuda'
        denoise (bool): Whether to apply denoising. Default: True
        margin (int): Margin for chunking in samples. Default: 44100
        chunks (int): Number of chunks to process at once. Default: 15
        n_fft (int): FFT size. Default: 6144
        dim_t (int): Time dimension. Default: 8
        dim_f (int): Frequency dimension. Default: 3072

    Returns:
        Manifest entries with audio files replaced by separated vocals.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.SourceSeparation
              model_path: "/models/UVR-MDX-NET-Inst_HQ_3.onnx"
              audio_filepath_key: audio_filepath
              output_dir: ${workspace_dir}/separated_audio
              device: cuda
    """

    def __init__(
        self,
        model_path: str,
        audio_filepath_key: str = "audio_filepath",
        output_dir: str = None,
        device: str = "cuda",
        denoise: bool = True,
        margin: int = 44100,
        chunks: int = 15,
        n_fft: int = 6144,
        dim_t: int = 8,
        dim_f: int = 3072,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.audio_filepath_key = audio_filepath_key
        self.output_dir = output_dir
        self.device = device
        self.denoise = denoise
        self.margin = margin
        self.chunks = chunks
        self.n_fft = n_fft
        self.dim_t = dim_t
        self.dim_f = dim_f
        self.predictor = None

    def prepare(self):
        """Initialize the source separation model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download from: https://github.com/TRvlvr/model_repo/releases/download/"
                "all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx"
            )
        
        logger.info(f"Loading source separation model from {self.model_path}")
        self.predictor = Predictor(
            model_path=self.model_path,
            device=self.device,
            denoise=self.denoise,
            margin=self.margin,
            chunks=self.chunks,
            n_fft=self.n_fft,
            dim_t=self.dim_t,
            dim_f=self.dim_f,
        )
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Separated audio will be saved to: {self.output_dir}")

    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Process a single audio file for source separation."""
        audio_filepath = data_entry[self.audio_filepath_key]
        
        try:
            # Load audio at 44.1kHz (required for the model)
            audio, sr = librosa.load(audio_filepath, sr=44100, mono=False)
            
            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio])
            
            # Perform source separation
            separated_vocals = self.predictor.predict(audio)
            
            # Determine output path
            if self.output_dir:
                basename = os.path.basename(audio_filepath)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(self.output_dir, f"{name}_separated{ext}")
            else:
                output_path = audio_filepath
            
            # Save separated audio
            sf.write(output_path, separated_vocals, 44100)
            
            # Update manifest entry
            data_entry[self.audio_filepath_key] = output_path
            
            return [DataEntry(data=data_entry)]
            
        except Exception as e:
            logger.error(f"Error processing {audio_filepath}: {e}")
            return [DataEntry(data=None)]

