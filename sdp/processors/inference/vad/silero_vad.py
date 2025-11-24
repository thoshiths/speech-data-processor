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
Silero VAD Segmentation Processor.

Based on Amphion's Emilia preprocessing pipeline.
Performs fine-grained voice activity detection and segmentation.
"""

import os
import librosa
import torch
import numpy as np
from typing import List
import pandas as pd

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest


class SileroVAD:
    """
    Voice Activity Detection (VAD) using Silero-VAD.
    """

    def __init__(self, device=torch.device("cpu")):
        """
        Initialize the VAD object.

        Args:
            device (torch.device): The device to run the model on.
        """
        try:
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
                source="github",
            )
            self.vad_model = vad_model
            (get_speech_timestamps, _, _, _, _) = utils
            self.get_speech_timestamps = get_speech_timestamps
            self.device = device
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {e}")

    def segment_speech(self, audio_segment, start_time, end_time, sampling_rate):
        """
        Segment speech from an audio segment and return a list of timestamps.

        Args:
            audio_segment (np.ndarray): The audio segment to be segmented.
            start_time (int): The start time of the audio segment in frames.
            end_time (int): The end time of the audio segment in frames.
            sampling_rate (int): The sampling rate of the audio segment.

        Returns:
            list: A list of timestamps, each containing the start and end times of speech segments in frames.
        """
        if audio_segment is None or not isinstance(audio_segment, (np.ndarray, list)):
            raise ValueError("Invalid audio segment")

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=sampling_rate
        )

        adjusted_timestamps = [
            (ts["start"] + start_time, ts["end"] + start_time)
            for ts in speech_timestamps
        ]
        if not adjusted_timestamps:
            return []

        intervals = [
            end[0] - start[1]
            for start, end in zip(adjusted_timestamps[:-1], adjusted_timestamps[1:])
        ]

        segments = []

        def split_timestamps(start_index, end_index):
            if (
                start_index == end_index
                or adjusted_timestamps[end_index][1]
                - adjusted_timestamps[start_index][0]
                < 20 * sampling_rate
            ):
                segments.append([start_index, end_index])
            else:
                if not intervals[start_index:end_index]:
                    return
                max_interval_index = intervals[start_index:end_index].index(
                    max(intervals[start_index:end_index])
                )
                split_index = start_index + max_interval_index
                split_timestamps(start_index, split_index)
                split_timestamps(split_index + 1, end_index)

        split_timestamps(0, len(adjusted_timestamps) - 1)

        merged_timestamps = [
            [adjusted_timestamps[start][0], adjusted_timestamps[end][1]]
            for start, end in segments
        ]
        return merged_timestamps

    def process_segments(self, diarization_segments, audio_data, sample_rate):
        """
        Process the audio based on the given speaker diarization segments.

        Args:
            diarization_segments (list): List of dicts with 'start', 'end', 'speaker' keys.
            audio_data (np.ndarray): The audio waveform.
            sample_rate (int): The audio sample rate.

        Returns:
            list: A list of dictionaries containing processed audio segments with start, end, and speaker.
        """
        VAD_THRESHOLD = 20  # seconds
        SAMPLING_RATE = 16000  # Silero VAD works at 16kHz
        
        out = []
        last_end = 0
        count_id = 0

        for segment in diarization_segments:
            start = float(segment["start"])
            end = float(segment["end"])

            if end <= last_end:
                continue
            last_end = end

            start_frame = int(start * sample_rate)
            end_frame = int(end * sample_rate)

            if end - start <= VAD_THRESHOLD:
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": start,
                        "end": end,
                        "speaker": segment.get("speaker", "SPEAKER_00"),
                    }
                )
                count_id += 1
                continue

            temp_audio = audio_data[start_frame:end_frame]

            # Resample to 16kHz for VAD
            temp_audio_resampled = librosa.resample(
                temp_audio, orig_sr=sample_rate, target_sr=SAMPLING_RATE
            )

            for start_frame_sub, end_frame_sub in self.segment_speech(
                temp_audio_resampled,
                int(start * SAMPLING_RATE),
                int(end * SAMPLING_RATE),
                SAMPLING_RATE,
            ):
                out.append(
                    {
                        "index": str(count_id).zfill(5),
                        "start": start_frame_sub / SAMPLING_RATE,
                        "end": end_frame_sub / SAMPLING_RATE,
                        "speaker": segment.get("speaker", "SPEAKER_00"),
                    }
                )
                count_id += 1

        return out


class SileroVADSegmentation(BaseProcessor):
    """
    Perform fine-grained VAD segmentation using Silero VAD.

    This processor takes speaker diarization output and performs fine-grained
    segmentation within each speaker segment using Silero VAD.

    Args:
        audio_filepath_key (str): Key for audio file path in manifest. Default: "audio_filepath"
        speaker_segments_key (str): Key for speaker diarization segments. Default: "speaker_segments"
        output_segments_key (str): Key to store VAD segments. Default: "vad_segments"
        min_segment_duration (float): Minimum segment duration in seconds. Default: 3.0
        max_segment_duration (float): Maximum segment duration in seconds. Default: 30.0
        merge_gap (float): Merge segments if gap is less than this (seconds). Default: 2.0
        device (str): Device to run VAD model ('cuda' or 'cpu'). Default: 'cpu'

    Returns:
        Manifest entries with fine-grained VAD segments added.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.SileroVADSegmentation
              audio_filepath_key: audio_filepath
              speaker_segments_key: speaker_segments
              output_segments_key: vad_segments
              min_segment_duration: 3.0
              max_segment_duration: 30.0
              device: cpu
    """

    def __init__(
        self,
        audio_filepath_key: str = "audio_filepath",
        speaker_segments_key: str = "speaker_segments",
        output_segments_key: str = "vad_segments",
        min_segment_duration: float = 3.0,
        max_segment_duration: float = 30.0,
        merge_gap: float = 2.0,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_filepath_key = audio_filepath_key
        self.speaker_segments_key = speaker_segments_key
        self.output_segments_key = output_segments_key
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.merge_gap = merge_gap
        self.device = torch.device(device)
        self.vad = None

    def prepare(self):
        """Initialize the Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        self.vad = SileroVAD(device=self.device)

    def process(self):
        """Process the manifest file."""
        self.prepare()
        
        input_entries = load_manifest(self.input_manifest_file)
        output_entries = []

        for entry in input_entries:
            audio_filepath = entry[self.audio_filepath_key]
            speaker_segments = entry.get(self.speaker_segments_key, [])

            if not speaker_segments:
                logger.warning(f"No speaker segments found for {audio_filepath}")
                continue

            try:
                # Load audio
                audio_data, sample_rate = librosa.load(audio_filepath, sr=None, mono=True)

                # Perform VAD segmentation
                vad_segments = self.vad.process_segments(
                    speaker_segments, audio_data, sample_rate
                )

                # Post-process: merge and filter segments
                vad_segments = self.cut_by_speaker_label(vad_segments)

                # Add segments to entry
                entry[self.output_segments_key] = vad_segments

                output_entries.append(entry)

            except Exception as e:
                logger.error(f"Error processing {audio_filepath}: {e}")
                continue

        save_manifest(self.output_manifest_file, output_entries)
        logger.info(f"Processed {len(output_entries)} entries with VAD segmentation")

    def cut_by_speaker_label(self, vad_list):
        """
        Merge and trim VAD segments by speaker labels.

        Args:
            vad_list (list): List of VAD segments with start, end, and speaker labels.

        Returns:
            list: A list of updated VAD segments after merging and trimming.
        """
        updated_list = []

        for idx, vad in enumerate(vad_list):
            last_start_time = updated_list[-1]["start"] if updated_list else None
            last_end_time = updated_list[-1]["end"] if updated_list else None
            last_speaker = updated_list[-1]["speaker"] if updated_list else None

            # Handle segments longer than max duration
            if vad["end"] - vad["start"] >= self.max_segment_duration:
                current_start = vad["start"]
                segment_end = vad["end"]
                logger.debug(
                    f"Segment longer than {self.max_segment_duration}s, splitting"
                )
                while segment_end - current_start >= self.max_segment_duration:
                    new_vad = vad.copy()
                    new_vad["end"] = current_start + self.max_segment_duration
                    new_vad["start"] = current_start
                    updated_list.append(new_vad)
                    current_start += self.max_segment_duration
                    vad = vad.copy()
                    vad["start"] = current_start
                    vad["end"] = segment_end
                updated_list.append(vad)
                continue

            # Append if different speaker or long enough
            if (
                last_speaker is None
                or last_speaker != vad["speaker"]
                or vad["end"] - vad["start"] >= self.min_segment_duration
            ):
                updated_list.append(vad)
                continue

            # Merge if gap is small and total duration is acceptable
            if (
                vad["start"] - last_end_time >= self.merge_gap
                or vad["end"] - last_start_time >= self.max_segment_duration
            ):
                updated_list.append(vad)
            else:
                updated_list[-1]["end"] = vad["end"]  # Merge

        # Filter by minimum duration
        filter_list = [
            vad
            for vad in updated_list
            if vad["end"] - vad["start"] >= self.min_segment_duration
        ]

        logger.info(
            f"VAD segmentation: {len(vad_list)} → {len(updated_list)} merged → "
            f"{len(filter_list)} after filtering"
        )

        return filter_list

