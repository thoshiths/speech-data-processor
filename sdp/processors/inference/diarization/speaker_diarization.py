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
Speaker Diarization Processor using PyAnnote.

Performs speaker diarization to identify different speakers in audio files.
Based on Amphion's Emilia preprocessing pipeline.
"""

import torch
import torchaudio
from pyannote.audio import Pipeline
import pandas as pd

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest


class SpeakerDiarization(BaseProcessor):
    """
    Perform speaker diarization using PyAnnote.

    This processor identifies different speakers in audio files and creates
    segments labeled by speaker.

    Args:
        hf_token (str): HuggingFace authentication token for accessing pretrained models.
            Get token at: https://huggingface.co/settings/tokens
            Grant access at: https://huggingface.co/pyannote/speaker-diarization-3.1
        audio_filepath_key (str): Key for audio file path in manifest. Default: "audio_filepath"
        output_segments_key (str): Key to store speaker segments. Default: "speaker_segments"
        segmentation_batch_size (int): Batch size for segmentation. Default: 128
        embedding_batch_size (int): Batch size for speaker embeddings. Default: 128
        device (str): Device to run the models on ('cuda' or 'cpu'). Default: "cuda"

    Returns:
        Manifest entries with speaker diarization segments added.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.SpeakerDiarization
              hf_token: ${hf_token}
              audio_filepath_key: audio_filepath
              output_segments_key: speaker_segments
              device: cuda
    """

    def __init__(
        self,
        hf_token: str,
        audio_filepath_key: str = "audio_filepath",
        output_segments_key: str = "speaker_segments",
        segmentation_batch_size: int = 128,
        embedding_batch_size: int = 128,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hf_token = hf_token
        self.audio_filepath_key = audio_filepath_key
        self.output_segments_key = output_segments_key
        self.segmentation_batch_size = segmentation_batch_size
        self.embedding_batch_size = embedding_batch_size
        self.device = torch.device(device)
        self.pipeline = None

    def prepare(self):
        """Initialize the speaker diarization pipeline."""
        if not self.hf_token.startswith("hf"):
            raise ValueError(
                "hf_token must start with 'hf'. Get your token at: "
                "https://huggingface.co/settings/tokens "
                "Remember to grant access at: "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        logger.info("Loading speaker diarization pipeline...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        self.pipeline.to(self.device)
        self.pipeline.segmentation_batch_size = self.segmentation_batch_size
        self.pipeline.embedding_batch_size = self.embedding_batch_size
        logger.info("Speaker diarization pipeline loaded successfully")

    def process(self):
        """Process the manifest file."""
        self.prepare()

        input_entries = load_manifest(self.input_manifest_file)
        output_entries = []

        for entry in input_entries:
            audio_filepath = entry[self.audio_filepath_key]

            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(audio_filepath)
                waveform = waveform.to(self.device)

                # Ensure mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Perform speaker diarization
                diarization = self.pipeline(
                    {
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                        "channel": 0,
                    }
                )

                # Convert to segments list
                segments = []
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    segments.append(
                        {
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker,
                        }
                    )

                # Add segments to entry
                entry[self.output_segments_key] = segments

                logger.info(
                    f"Diarized {audio_filepath}: {len(segments)} segments, "
                    f"{len(set(s['speaker'] for s in segments))} speakers"
                )

                output_entries.append(entry)

            except Exception as e:
                logger.error(f"Error processing {audio_filepath}: {e}")
                continue

        save_manifest(self.output_manifest_file, output_entries)
        logger.info(
            f"Processed {len(output_entries)} entries with speaker diarization"
        )

