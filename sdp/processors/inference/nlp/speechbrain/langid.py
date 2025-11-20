# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
SpeechBrain VoxLingua107 Language Identification Processor

This processor uses SpeechBrain's pre-trained VoxLingua107 ECAPA model
for audio-based language identification.
"""

from typing import List, Optional
import os

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.logging import logger


class SpeechBrainLangId(BaseParallelProcessor):
    """
    Language identification using SpeechBrain's VoxLingua107 ECAPA model.
    
    This processor detects the language of audio files using a pre-trained
    ECAPA-TDNN model trained on VoxLingua107 dataset (107 languages).
    
    Args:
        input_audio_key (str): Key for input audio filepath.
        output_lang_key (str): Key to store detected language code.
        output_confidence_key (str): Key to store confidence score.
        model_source (str): HuggingFace model source.
            Default: "speechbrain/lang-id-voxlingua107-ecapa"
        save_dir (str): Directory to cache the model.
            Default: "tmp/speechbrain_langid"
        min_confidence (float): Minimum confidence threshold (0-1).
            Samples below this are marked as "unknown".
            Default: 0.5
        device (str): Device to run inference on ("cuda" or "cpu").
            Default: "cuda"
        **kwargs: Additional arguments for BaseParallelProcessor.
    
    Returns:
        Manifest entries with added language detection fields:
        - output_lang_key: ISO language code (e.g., 'hi', 'en', 'te')
        - output_confidence_key: Confidence score (0-1)
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.SpeechBrainLangId
              input_audio_key: audio_filepath
              output_lang_key: detected_lang
              output_confidence_key: lang_confidence
              min_confidence: 0.7
              device: cuda
    
    Note:
        Requires: pip install speechbrain torchaudio
    """
    
    def __init__(
        self,
        input_audio_key: str = "audio_filepath",
        output_lang_key: str = "detected_lang",
        output_confidence_key: str = "lang_confidence",
        model_source: str = "speechbrain/lang-id-voxlingua107-ecapa",
        save_dir: str = "tmp/speechbrain_langid",
        min_confidence: float = 0.5,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_key = input_audio_key
        self.output_lang_key = output_lang_key
        self.output_confidence_key = output_confidence_key
        self.model_source = model_source
        self.save_dir = save_dir
        self.min_confidence = min_confidence
        self.device = device
        self.language_id = None
        
    def prepare(self):
        """Load the SpeechBrain language ID model."""
        try:
            import torch
            import torchaudio
            
            # Fix for torchaudio 2.1.0+ compatibility with speechbrain
            # 1. Set backend if not already set (suppresses warning)
            try:
                if hasattr(torchaudio, 'set_audio_backend'):
                    # For torchaudio < 2.1.0
                    torchaudio.set_audio_backend("soundfile")
                elif hasattr(torchaudio, 'get_audio_backend'):
                    # For torchaudio >= 2.1.0, backend is automatic
                    pass
            except:
                pass
            
            # 2. Monkey-patch list_audio_backends if it doesn't exist
            if not hasattr(torchaudio, 'list_audio_backends'):
                torchaudio.list_audio_backends = lambda: ['soundfile', 'sox_io']
            
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as e:
            raise ImportError(
                "SpeechBrain or torchaudio is not installed. "
                "Please install with: pip install speechbrain torchaudio soundfile\n"
                f"Error: {e}"
            )
        
        logger.info(f"Loading SpeechBrain language ID model from {self.model_source}")
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load model
        self.language_id = EncoderClassifier.from_hparams(
            source=self.model_source,
            savedir=self.save_dir,
            run_opts={"device": self.device}
        )
        
        logger.info("SpeechBrain language ID model loaded successfully")
    
    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Process a single audio file for language identification."""
        audio_filepath = data_entry[self.input_audio_key]
        
        try:
            # Load audio
            signal = self.language_id.load_audio(audio_filepath)
            
            # Perform language identification
            prediction = self.language_id.classify_batch(signal)
            
            # Extract results
            # prediction is a tuple: (log_probs, confidence, index, [language_code])
            confidence = prediction[1].exp().item()  # Convert log to linear scale
            lang_code = prediction[3][0]  # Get language code
            
            # Clean language code (remove description if present)
            # E.g., "th: Thai" -> "th"
            if ":" in lang_code:
                lang_code = lang_code.split(":")[0].strip()
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                logger.debug(
                    f"Low confidence ({confidence:.3f}) for {audio_filepath}, "
                    f"marking as unknown"
                )
                lang_code = "unknown"
            
            # Add results to data entry
            data_entry[self.output_lang_key] = lang_code
            data_entry[self.output_confidence_key] = round(confidence, 4)
            
            logger.debug(
                f"Detected language: {lang_code} "
                f"(confidence: {confidence:.3f}) for {audio_filepath}"
            )
            
        except Exception as e:
            logger.warning(f"Error processing {audio_filepath}: {e}")
            data_entry[self.output_lang_key] = "error"
            data_entry[self.output_confidence_key] = 0.0
        
        return [DataEntry(data=data_entry)]


class WhisperLangId(BaseParallelProcessor):
    """
    Language identification using Faster Whisper's built-in language detection.
    
    This processor uses Faster Whisper Large V3 model's language detection
    capability to identify the language of audio files.
    
    Args:
        input_audio_key (str): Key for input audio filepath.
        output_lang_key (str): Key to store detected language code.
        output_confidence_key (str): Key to store confidence score.
        model_size (str): Whisper model size.
            Options: "large-v3", "large-v2", "medium", "small", "base", "tiny"
            Default: "large-v3"
        device (str): Device to run inference on ("cuda" or "cpu").
            Default: "cuda"
        compute_type (str): Compute precision.
            Options: "float16", "int8", "float32"
            Default: "float16"
        min_confidence (float): Minimum confidence threshold (0-1).
            Default: 0.5
        **kwargs: Additional arguments for BaseParallelProcessor.
    
    Returns:
        Manifest entries with added language detection fields:
        - output_lang_key: ISO language code
        - output_confidence_key: Confidence score (0-1)
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.WhisperLangId
              input_audio_key: audio_filepath
              output_lang_key: detected_lang
              output_confidence_key: lang_confidence
              model_size: "large-v3"
              device: cuda
    
    Note:
        Requires: pip install faster-whisper
    """
    
    def __init__(
        self,
        input_audio_key: str = "audio_filepath",
        output_lang_key: str = "detected_lang",
        output_confidence_key: str = "lang_confidence",
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        min_confidence: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_key = input_audio_key
        self.output_lang_key = output_lang_key
        self.output_confidence_key = output_confidence_key
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.min_confidence = min_confidence
        self.model = None
        
    def prepare(self):
        """Load the Faster Whisper model."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "Faster Whisper is not installed. "
                "Please install it with: pip install faster-whisper"
            )
        
        logger.info(f"Loading Faster Whisper {self.model_size} model")
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        
        logger.info("Faster Whisper model loaded successfully")
    
    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Process a single audio file for language identification."""
        audio_filepath = data_entry[self.input_audio_key]
        
        try:
            # Detect language using first 30 seconds
            audio_info = self.model.detect_language(audio_filepath)
            
            # audio_info is a tuple: (language_code, confidence)
            lang_code = audio_info[0]
            confidence = audio_info[1]
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                logger.debug(
                    f"Low confidence ({confidence:.3f}) for {audio_filepath}, "
                    f"marking as unknown"
                )
                lang_code = "unknown"
            
            # Add results to data entry
            data_entry[self.output_lang_key] = lang_code
            data_entry[self.output_confidence_key] = round(confidence, 4)
            
            logger.debug(
                f"Detected language: {lang_code} "
                f"(confidence: {confidence:.3f}) for {audio_filepath}"
            )
            
        except Exception as e:
            logger.warning(f"Error processing {audio_filepath}: {e}")
            data_entry[self.output_lang_key] = "error"
            data_entry[self.output_confidence_key] = 0.0
        
        return [DataEntry(data=data_entry)]


class CrossValidateLangId(BaseParallelProcessor):
    """
    Cross-validate language identification using multiple models.
    
    This processor runs multiple language ID models and uses voting
    or confidence-weighted consensus to determine the final language.
    
    Args:
        input_audio_key (str): Key for input audio filepath.
        output_lang_key (str): Key to store final detected language.
        output_confidence_key (str): Key to store final confidence.
        lang_keys (list): List of language field keys from different models.
            E.g., ["nemo_lang", "speechbrain_lang", "whisper_lang"]
        confidence_keys (list): List of confidence field keys from different models.
            E.g., ["nemo_conf", "speechbrain_conf", "whisper_conf"]
        method (str): Consensus method.
            Options: "voting" (majority vote), "confidence" (highest confidence),
                    "weighted" (confidence-weighted voting)
            Default: "weighted"
        require_agreement (int): Minimum number of models that must agree.
            If not met, result is marked as "uncertain".
            Default: 2
        **kwargs: Additional arguments for BaseParallelProcessor.
    
    Returns:
        Manifest entries with final consensus language and confidence.
    
    Example:
        .. code-block:: yaml
        
            # First run multiple LangID models
            - _target_: sdp.processors.AudioLid
              output_lang_key: nemo_lang
              output_confidence_key: nemo_conf
            
            - _target_: sdp.processors.SpeechBrainLangId
              output_lang_key: speechbrain_lang
              output_confidence_key: speechbrain_conf
            
            - _target_: sdp.processors.WhisperLangId
              output_lang_key: whisper_lang
              output_confidence_key: whisper_conf
            
            # Then cross-validate
            - _target_: sdp.processors.CrossValidateLangId
              lang_keys: ["nemo_lang", "speechbrain_lang", "whisper_lang"]
              confidence_keys: ["nemo_conf", "speechbrain_conf", "whisper_conf"]
              method: "weighted"
              require_agreement: 2
    """
    
    def __init__(
        self,
        input_audio_key: str = "audio_filepath",
        output_lang_key: str = "final_lang",
        output_confidence_key: str = "final_confidence",
        lang_keys: List[str] = None,
        confidence_keys: List[str] = None,
        method: str = "weighted",
        require_agreement: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_audio_key = input_audio_key
        self.output_lang_key = output_lang_key
        self.output_confidence_key = output_confidence_key
        self.lang_keys = lang_keys or []
        self.confidence_keys = confidence_keys or []
        self.method = method
        self.require_agreement = require_agreement
        
        if len(self.lang_keys) != len(self.confidence_keys):
            raise ValueError("lang_keys and confidence_keys must have same length")
    
    def process_dataset_entry(self, data_entry) -> List[DataEntry]:
        """Cross-validate language predictions from multiple models."""
        from collections import Counter
        
        predictions = []
        
        # Collect predictions from all models
        for lang_key, conf_key in zip(self.lang_keys, self.confidence_keys):
            if lang_key in data_entry and conf_key in data_entry:
                lang = data_entry[lang_key]
                conf = data_entry[conf_key]
                
                # Skip errors and unknowns
                if lang not in ["error", "unknown"]:
                    predictions.append((lang, conf))
        
        if not predictions:
            data_entry[self.output_lang_key] = "unknown"
            data_entry[self.output_confidence_key] = 0.0
            return [DataEntry(data=data_entry)]
        
        # Determine consensus based on method
        if self.method == "voting":
            # Simple majority voting
            votes = Counter(lang for lang, _ in predictions)
            final_lang = votes.most_common(1)[0][0]
            agreement_count = votes[final_lang]
            
            # Average confidence for voted language
            final_conf = sum(conf for lang, conf in predictions if lang == final_lang) / agreement_count
            
        elif self.method == "confidence":
            # Highest confidence wins
            final_lang, final_conf = max(predictions, key=lambda x: x[1])
            agreement_count = sum(1 for lang, _ in predictions if lang == final_lang)
            
        elif self.method == "weighted":
            # Confidence-weighted voting
            lang_scores = {}
            for lang, conf in predictions:
                lang_scores[lang] = lang_scores.get(lang, 0) + conf
            
            final_lang = max(lang_scores, key=lang_scores.get)
            final_conf = lang_scores[final_lang] / len(predictions)
            agreement_count = sum(1 for lang, _ in predictions if lang == final_lang)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Check if minimum agreement is met
        if agreement_count < self.require_agreement:
            logger.debug(
                f"Insufficient agreement ({agreement_count}/{len(predictions)}) "
                f"for {data_entry.get(self.input_audio_key, 'unknown')}"
            )
            data_entry[self.output_lang_key] = "uncertain"
            data_entry[self.output_confidence_key] = 0.0
        else:
            data_entry[self.output_lang_key] = final_lang
            data_entry[self.output_confidence_key] = round(final_conf, 4)
        
        # Store agreement info
        data_entry["lang_agreement"] = agreement_count
        data_entry["lang_total_models"] = len(predictions)
        
        return [DataEntry(data=data_entry)]

