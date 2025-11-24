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
Whisper Language Detection Processor (Per-Segment, No ASR).

Based on Amphion's Emilia preprocessing pipeline.
Uses Whisper only for language detection on individual segments,
then routes to language-specific NeMo ASR models.
"""

import os
import numpy as np
import librosa
import torch
from faster_whisper import WhisperModel
from typing import List

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest


class WhisperLanguageDetector:
    """
    Language detector using Faster Whisper (no ASR).
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        device_index: int = 0,
    ):
        """
        Initialize Whisper language detector.

        Args:
            model_size (str): Whisper model size
            device (str): Device ('cuda' or 'cpu')
            compute_type (str): Compute type ('float16', 'int8', 'float32')
            device_index (int): GPU device index
        """
        logger.info(f"Loading Whisper {model_size} for language detection...")
        
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            device_index=device_index,
        )
        
        self.model_size = model_size
        self.device = device
        logger.info(f"Whisper {model_size} loaded successfully on {device}")

    def detect_language(self, audio_array: np.ndarray, sample_rate: int = 16000):
        """
        Detect language from audio segment.

        Args:
            audio_array (np.ndarray): Audio array at 16kHz
            sample_rate (int): Sample rate (must be 16000 for Whisper)

        Returns:
            tuple: (language_code, probability)
        """
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_array = librosa.resample(
                audio_array, orig_sr=sample_rate, target_sr=16000
            )

        # Detect language
        try:
            language_info = self.model.detect_language(audio_array)
            # language_info is a tuple: (language_code, probability)
            language_code = language_info[0]
            probability = language_info[1]
            
            return language_code, probability
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return None, 0.0


class WhisperSegmentLanguageDetection(BaseProcessor):
    """
    Detect language for each segment using Whisper (no ASR).

    This processor takes VAD-segmented audio and detects the language
    of each segment using Whisper's language detection capability.
    It does NOT perform ASR - only language detection.

    Args:
        audio_filepath_key (str): Key for audio file path in manifest. Default: "audio_filepath"
        segments_key (str): Key for segments list in manifest. Default: "vad_segments"
        output_lang_key (str): Key to store detected language. Default: "detected_lang"
        output_confidence_key (str): Key to store confidence score. Default: "lang_confidence"
        model_size (str): Whisper model size. Default: "large-v3"
            Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
        device (str): Device to run model ('cuda' or 'cpu'). Default: "cpu"
        compute_type (str): Compute type. Default: "int8"
            Options: "float16" (CUDA only), "int8", "float32"
        min_confidence (float): Minimum confidence threshold (0-1). Default: 0.8
        supported_languages (list): List of supported language codes. Default: None (all languages)
        
    Returns:
        Manifest entries with language detected for each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.WhisperSegmentLanguageDetection
              audio_filepath_key: audio_filepath
              segments_key: vad_segments
              output_lang_key: detected_lang
              output_confidence_key: lang_confidence
              model_size: "large-v3"
              device: cpu
              compute_type: int8
              min_confidence: 0.8
              supported_languages: ["en", "hi", "te", "ta", "bn"]
    """

    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        segments_key: str = "vad_segments",
        output_lang_key: str = "detected_lang",
        output_confidence_key: str = "lang_confidence",
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        min_confidence: float = 0.8,
        supported_languages: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.segments_key = segments_key
        self.output_lang_key = output_lang_key
        self.output_confidence_key = output_confidence_key
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.min_confidence = min_confidence
        self.supported_languages = supported_languages
        self.detector = None

    def prepare(self):
        """Initialize the Whisper language detector."""
        self.detector = WhisperLanguageDetector(
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def process(self):
        """Process the manifest file."""
        self.prepare()

        input_entries = load_manifest(self.input_manifest_file)
        output_entries = []
        
        total_segments = 0
        valid_segments = 0
        filtered_segments = 0

        for entry in input_entries:
            audio_filepath = entry[self.audio_filepath_key]
            segments = entry.get(self.segments_key, [])

            if not segments:
                logger.warning(f"No segments found for {audio_filepath}")
                continue

            try:
                # Load audio
                audio_data, sample_rate = librosa.load(audio_filepath, sr=16000, mono=True)

                # Detect language for each segment
                updated_segments = []
                for segment in segments:
                    total_segments += 1
                    start = segment["start"]
                    end = segment["end"]
                    
                    # Extract segment audio
                    start_frame = int(start * sample_rate)
                    end_frame = int(end * sample_rate)
                    segment_audio = audio_data[start_frame:end_frame]

                    # Detect language
                    language, probability = self.detector.detect_language(
                        segment_audio, sample_rate
                    )

                    # Filter by confidence and supported languages
                    if language is None or probability < self.min_confidence:
                        filtered_segments += 1
                        logger.debug(
                            f"Segment filtered: lang={language}, prob={probability:.2f}"
                        )
                        continue

                    if (
                        self.supported_languages
                        and language not in self.supported_languages
                    ):
                        filtered_segments += 1
                        logger.debug(
                            f"Segment filtered: unsupported language {language}"
                        )
                        continue

                    # Add language info to segment
                    segment[self.output_lang_key] = language
                    segment[self.output_confidence_key] = probability
                    updated_segments.append(segment)
                    valid_segments += 1

                if updated_segments:
                    entry[self.segments_key] = updated_segments
                    output_entries.append(entry)
                    
                    # Log language distribution for this file
                    lang_counts = {}
                    for seg in updated_segments:
                        lang = seg[self.output_lang_key]
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                    
                    logger.info(
                        f"{audio_filepath}: {len(updated_segments)} segments, "
                        f"languages: {lang_counts}"
                    )

            except Exception as e:
                logger.error(f"Error processing {audio_filepath}: {e}")
                continue

        save_manifest(self.output_manifest_file, output_entries)
        
        logger.info(
            f"Language detection complete: {total_segments} total segments, "
            f"{valid_segments} valid, {filtered_segments} filtered "
            f"({filtered_segments/total_segments*100:.1f}% filtered)"
        )
        logger.info(f"Processed {len(output_entries)} audio files")

