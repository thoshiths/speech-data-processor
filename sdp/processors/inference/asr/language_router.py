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
Language-Based ASR Router Processor.

Routes segments to language-specific NeMo ASR models based on detected language.
Performs batched ASR for efficiency.
"""

import os
import librosa
import soundfile as sf
import tempfile
from collections import defaultdict
from typing import Dict, List
import nemo.collections.asr as nemo_asr
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest


class LanguageBasedASRRouter(BaseProcessor):
    """
    Route segments to language-specific ASR models based on detected language.

    This processor takes segments with detected languages and routes them to
    the appropriate NeMo ASR model for each language. It performs batched
    transcription for efficiency.

    Args:
        audio_filepath_key (str): Key for audio file path. Default: "audio_filepath"
        segments_key (str): Key for segments with language info. Default: "vad_segments"
        lang_key (str): Key for detected language in segments. Default: "detected_lang"
        output_text_key (str): Key to store transcription. Default: "text"
        language_models (dict): Mapping of language codes to NeMo ASR model paths.
            Example: {"en": "/path/to/english.nemo", "hi": "/path/to/hindi.nemo"}
        batch_size (int): Batch size for ASR inference. Default: 32
        output_audio_dir (str): Directory to save segment audio files. Default: None (temp dir)
        device (str): Device to run ASR models ('cuda' or 'cpu'). Default: "cuda"

    Returns:
        Manifest entries with transcriptions added to each segment.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.LanguageBasedASRRouter
              audio_filepath_key: audio_filepath
              segments_key: vad_segments
              lang_key: detected_lang
              output_text_key: text
              language_models:
                en: /models/english.nemo
                hi: /models/hindi.nemo
                te: /models/telugu.nemo
                ta: /models/tamil.nemo
              batch_size: 32
              device: cuda
    """

    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        segments_key: str = "vad_segments",
        lang_key: str = "detected_lang",
        output_text_key: str = "text",
        language_models: Dict[str, str] = None,
        batch_size: int = 32,
        output_audio_dir: str = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.segments_key = segments_key
        self.lang_key = lang_key
        self.output_text_key = output_text_key
        self.language_models = language_models or {}
        self.batch_size = batch_size
        self.output_audio_dir = output_audio_dir
        self.device = device
        self.asr_models = {}
        self.temp_dir = None

    def prepare(self):
        """Load all ASR models."""
        if not self.language_models:
            raise ValueError("No language models provided in language_models parameter")

        logger.info(f"Loading ASR models for {len(self.language_models)} languages...")

        for lang, model_path in self.language_models.items():
            if not os.path.exists(model_path):
                logger.warning(f"Model not found for {lang}: {model_path}")
                continue

            try:
                logger.info(f"Loading {lang} ASR model from {model_path}")
                asr_model = nemo_asr.models.ASRModel.restore_from(model_path, map_location=self.device)
                asr_model.eval()
                if self.device == "cuda":
                    asr_model = asr_model.cuda()
                self.asr_models[lang] = asr_model
                logger.info(f"Loaded {lang} ASR model successfully")
            except Exception as e:
                logger.error(f"Failed to load {lang} ASR model: {e}")
                continue

        if not self.asr_models:
            raise RuntimeError("No ASR models loaded successfully")

        # Create temp directory for segment audio if needed
        if not self.output_audio_dir:
            self.temp_dir = tempfile.mkdtemp()
            self.output_audio_dir = self.temp_dir
            logger.info(f"Using temporary directory for segments: {self.temp_dir}")
        else:
            os.makedirs(self.output_audio_dir, exist_ok=True)

    def process(self):
        """Process the manifest file."""
        self.prepare()

        input_entries = load_manifest(self.input_manifest_file)
        
        # Group all segments by language across all files
        lang_segments = defaultdict(list)
        segment_to_entry = {}  # Map segment to its parent entry

        logger.info("Grouping segments by language...")
        for entry_idx, entry in enumerate(input_entries):
            audio_filepath = entry[self.audio_filepath_key]
            segments = entry.get(self.segments_key, [])

            if not segments:
                continue

            try:
                # Load audio once per file
                audio_data, sample_rate = librosa.load(audio_filepath, sr=None, mono=True)

                for seg_idx, segment in enumerate(segments):
                    lang = segment.get(self.lang_key)
                    if not lang or lang not in self.asr_models:
                        continue

                    # Extract segment audio
                    start_frame = int(segment["start"] * sample_rate)
                    end_frame = int(segment["end"] * sample_rate)
                    segment_audio = audio_data[start_frame:end_frame]

                    # Save segment audio to temp file
                    segment_id = f"{entry_idx:06d}_{seg_idx:06d}"
                    segment_path = os.path.join(
                        self.output_audio_dir, f"{segment_id}_{lang}.wav"
                    )
                    sf.write(segment_path, segment_audio, sample_rate)

                    # Store segment info
                    segment_info = {
                        "path": segment_path,
                        "segment": segment,
                        "entry_idx": entry_idx,
                        "seg_idx": seg_idx,
                    }
                    lang_segments[lang].append(segment_info)
                    segment_to_entry[(entry_idx, seg_idx)] = segment

            except Exception as e:
                logger.error(f"Error processing {audio_filepath}: {e}")
                continue

        # Transcribe segments by language in batches
        logger.info("Performing language-specific ASR...")
        for lang, segments in lang_segments.items():
            if lang not in self.asr_models:
                logger.warning(f"No ASR model for language: {lang}")
                continue

            logger.info(f"Transcribing {len(segments)} {lang} segments...")
            asr_model = self.asr_models[lang]

            # Process in batches
            for i in tqdm(range(0, len(segments), self.batch_size), desc=f"ASR ({lang})"):
                batch = segments[i : i + self.batch_size]
                batch_paths = [seg["path"] for seg in batch]

                try:
                    # Perform batch ASR
                    transcriptions = asr_model.transcribe(batch_paths, batch_size=self.batch_size)
                    
                    # Add transcriptions to segments
                    for seg_info, transcription in zip(batch, transcriptions):
                        seg_info["segment"][self.output_text_key] = transcription
                        
                except Exception as e:
                    logger.error(f"Error transcribing batch for {lang}: {e}")
                    continue

        # Clean up temp files
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Cleaned up temporary segment files")

        # Save output manifest
        save_manifest(self.output_manifest_file, input_entries)
        
        # Log statistics
        total_segments = sum(len(segs) for segs in lang_segments.values())
        lang_stats = {lang: len(segs) for lang, segs in lang_segments.items()}
        logger.info(f"ASR complete: {total_segments} segments transcribed")
        logger.info(f"Language distribution: {lang_stats}")
        logger.info(f"Processed {len(input_entries)} audio files")


class LanguageBasedSegmentSplitter(BaseProcessor):
    """
    Split segments into individual manifest entries for language-specific processing.

    This processor takes manifest entries with multiple segments and creates
    separate manifest entries for each segment, preserving language information.
    Useful for routing to language-specific ASR processors.

    Args:
        audio_filepath_key (str): Key for audio file path. Default: "audio_filepath"
        segments_key (str): Key for segments list. Default: "vad_segments"
        lang_key (str): Key for language in segments. Default: "detected_lang"
        output_audio_dir (str): Directory to save individual segment audio files.
        output_lang_key (str): Key to store language in output. Default: "language"

    Returns:
        One manifest entry per segment with individual audio file.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.LanguageBasedSegmentSplitter
              segments_key: vad_segments
              lang_key: detected_lang
              output_audio_dir: ${workspace_dir}/segments
              output_lang_key: language
    """

    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        segments_key: str = "vad_segments",
        lang_key: str = "detected_lang",
        output_audio_dir: str = None,
        output_lang_key: str = "language",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.segments_key = segments_key
        self.lang_key = lang_key
        self.output_audio_dir = output_audio_dir
        self.output_lang_key = output_lang_key

    def process(self):
        """Process the manifest file."""
        if not self.output_audio_dir:
            raise ValueError("output_audio_dir must be specified")

        os.makedirs(self.output_audio_dir, exist_ok=True)

        input_entries = load_manifest(self.input_manifest_file)
        output_entries = []

        for entry_idx, entry in enumerate(input_entries):
            audio_filepath = entry[self.audio_filepath_key]
            segments = entry.get(self.segments_key, [])

            if not segments:
                continue

            try:
                # Load audio
                audio_data, sample_rate = librosa.load(audio_filepath, sr=None, mono=True)

                for seg_idx, segment in enumerate(segments):
                    # Extract segment audio
                    start_frame = int(segment["start"] * sample_rate)
                    end_frame = int(segment["end"] * sample_rate)
                    segment_audio = audio_data[start_frame:end_frame]

                    # Save segment audio
                    lang = segment.get(self.lang_key, "unknown")
                    segment_filename = f"{entry_idx:06d}_{seg_idx:06d}_{lang}.wav"
                    segment_path = os.path.join(self.output_audio_dir, segment_filename)
                    sf.write(segment_path, segment_audio, sample_rate)

                    # Create new manifest entry for this segment
                    new_entry = {
                        self.audio_filepath_key: segment_path,
                        "duration": segment["end"] - segment["start"],
                        self.output_lang_key: lang,
                        "start": segment["start"],
                        "end": segment["end"],
                        "source_file": audio_filepath,
                        "speaker": segment.get("speaker", "unknown"),
                    }
                    
                    # Copy other segment fields
                    for key, value in segment.items():
                        if key not in ["start", "end", "speaker", self.lang_key]:
                            new_entry[key] = value

                    output_entries.append(new_entry)

            except Exception as e:
                logger.error(f"Error processing {audio_filepath}: {e}")
                continue

        save_manifest(self.output_manifest_file, output_entries)
        logger.info(f"Split {len(input_entries)} files into {len(output_entries)} segments")

