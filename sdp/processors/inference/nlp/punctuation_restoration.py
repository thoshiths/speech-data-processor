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
Punctuation restoration processors using LLMs or traditional models.
"""

import json
from pathlib import Path
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import load_manifest, save_manifest


class LLMPunctuationRestoration(BaseProcessor):
    """Add punctuation using vLLM (supports any HuggingFace LLM model).
    
    This processor creates duplicate entries:
    - Original entry with "pnc": "no" (no punctuation)
    - New entry with "pnc": "yes" (with punctuation from LLM)
    
    Supports multilingual models for Indic languages.
    
    Args:
        input_text_key (str): Key for input text without punctuation. Defaults to "text"
        output_text_key (str): Key for output text with punctuation. Defaults to "text_pnc"
        pnc_flag_key (str): Key for punctuation flag. Defaults to "pnc"
        model_name (str): HuggingFace model name. Examples:
            - "meta-llama/Llama-3.1-8B-Instruct"
            - "google/gemma-2-9b-it"
            - "Qwen/Qwen2.5-7B-Instruct"
            - "ai4bharat/Airavata" (for Indic languages)
        prompt_template (str): Prompt template with {text} placeholder
        device (str): Device to run on. Defaults to "cuda"
        tensor_parallel_size (int): Number of GPUs for model parallelism. Defaults to 1
        max_tokens (int): Maximum tokens to generate. Defaults to 512
        temperature (float): Sampling temperature. Defaults to 0.0 (greedy)
        keep_original (bool): Keep original entry without punctuation. Defaults to True
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.LLMPunctuationRestoration
              output_manifest_file: ${manifest_dir}/with_punctuation.json
              input_text_key: text
              output_text_key: text_pnc
              model_name: "Qwen/Qwen2.5-7B-Instruct"
              prompt_template: "Add punctuation to this text: {text}"
              keep_original: true
    """
    
    def __init__(
        self,
        input_text_key: str = "text",
        output_text_key: str = "text_pnc",
        pnc_flag_key: str = "pnc",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt_template: str = None,
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        keep_original: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.pnc_flag_key = pnc_flag_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.keep_original = keep_original
        
        # Default prompt template
        if prompt_template is None:
            self.prompt_template = (
                "Add proper punctuation (periods, commas, question marks, etc.) "
                "and capitalization to the following text. "
                "Return ONLY the punctuated text without any explanation.\n\n"
                "Text: {text}\n\n"
                "Punctuated text:"
            )
        else:
            self.prompt_template = prompt_template
        
        logger.info(f"Initializing LLM punctuation restoration with model: {model_name}")
    
    def process(self):
        """Process manifest and add punctuation using LLM."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM is required for LLM punctuation restoration. "
                "Install with: pip install vllm"
            )
        
        # Load manifest
        manifest = load_manifest(self.input_manifest_file)
        
        logger.info(f"Loading LLM model: {self.model_name}")
        llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )
        
        # Prepare sampling parameters
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.95 if self.temperature > 0 else 1.0,
        )
        
        # Prepare prompts
        texts_to_process = []
        valid_entries = []
        
        for entry in manifest:
            text = entry.get(self.input_text_key, "")
            if text and text.strip():
                texts_to_process.append(self.prompt_template.format(text=text))
                valid_entries.append(entry)
        
        logger.info(f"Processing {len(texts_to_process)} texts for punctuation restoration")
        
        # Run LLM inference
        outputs = llm.generate(texts_to_process, sampling_params)
        
        # Create output manifest
        result_entries = []
        
        for entry, output in tqdm(zip(valid_entries, outputs), total=len(valid_entries), 
                                   desc="Adding punctuation"):
            # Keep original entry without punctuation
            if self.keep_original:
                original_entry = entry.copy()
                original_entry[self.pnc_flag_key] = "no"
                result_entries.append(original_entry)
            
            # Create punctuated entry
            punctuated_entry = entry.copy()
            punctuated_text = output.outputs[0].text.strip()
            punctuated_entry[self.output_text_key] = punctuated_text
            punctuated_entry[self.pnc_flag_key] = "yes"
            result_entries.append(punctuated_entry)
        
        # Save results
        save_manifest(result_entries, self.output_manifest_file)
        logger.info(f"Saved {len(result_entries)} entries ({len(valid_entries)} punctuated + "
                   f"{len(valid_entries) if self.keep_original else 0} original)")


class NeMoPunctuationRestoration(BaseProcessor):
    """Add punctuation using NeMo Punctuation & Capitalization model.
    
    Faster than LLM but less flexible. Good for English and some supported languages.
    Creates duplicate entries: one without punctuation (pnc=no) and one with (pnc=yes).
    
    Args:
        input_text_key (str): Key for input text. Defaults to "text"
        output_text_key (str): Key for output text with punctuation. Defaults to "text_pnc"
        pnc_flag_key (str): Key for punctuation flag. Defaults to "pnc"
        model_name (str): NeMo model name. Options:
            - "punctuation_en_bert" (English)
            - "punctuation_en_distilbert" (English, faster)
        model_path (str): Path to local NeMo model file. Overrides model_name
        batch_size (int): Batch size for processing. Defaults to 32
        device (str): Device to run on. Defaults to "cuda"
        keep_original (bool): Keep original entry without punctuation. Defaults to True
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.NeMoPunctuationRestoration
              output_manifest_file: ${manifest_dir}/with_punctuation.json
              input_text_key: text
              model_name: "punctuation_en_bert"
              batch_size: 64
    """
    
    def __init__(
        self,
        input_text_key: str = "text",
        output_text_key: str = "text_pnc",
        pnc_flag_key: str = "pnc",
        model_name: str = "punctuation_en_bert",
        model_path: str = None,
        batch_size: int = 32,
        device: str = "cuda",
        keep_original: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_text_key = input_text_key
        self.output_text_key = output_text_key
        self.pnc_flag_key = pnc_flag_key
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.keep_original = keep_original
    
    def process(self):
        """Process manifest and add punctuation using NeMo model."""
        try:
            from nemo.collections.nlp.models import PunctuationCapitalizationModel
            import torch
        except ImportError:
            raise ImportError(
                "NeMo is required for NeMo punctuation restoration. "
                "Install with: pip install nemo_toolkit[nlp]"
            )
        
        # Load model
        logger.info(f"Loading NeMo punctuation model: {self.model_name or self.model_path}")
        if self.model_path:
            model = PunctuationCapitalizationModel.restore_from(self.model_path)
        else:
            model = PunctuationCapitalizationModel.from_pretrained(self.model_name)
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        
        # Load manifest
        manifest = load_manifest(self.input_manifest_file)
        
        # Extract texts
        texts = []
        valid_entries = []
        for entry in manifest:
            text = entry.get(self.input_text_key, "")
            if text and text.strip():
                texts.append(text)
                valid_entries.append(entry)
        
        logger.info(f"Processing {len(texts)} texts for punctuation restoration")
        
        # Process in batches
        punctuated_texts = model.add_punctuation_capitalization(
            texts,
            batch_size=self.batch_size,
        )
        
        # Create output manifest
        result_entries = []
        
        for entry, punctuated_text in zip(valid_entries, punctuated_texts):
            # Keep original entry without punctuation
            if self.keep_original:
                original_entry = entry.copy()
                original_entry[self.pnc_flag_key] = "no"
                result_entries.append(original_entry)
            
            # Create punctuated entry
            punctuated_entry = entry.copy()
            punctuated_entry[self.output_text_key] = punctuated_text
            punctuated_entry[self.pnc_flag_key] = "yes"
            result_entries.append(punctuated_entry)
        
        # Save results
        save_manifest(result_entries, self.output_manifest_file)
        logger.info(f"Saved {len(result_entries)} entries ({len(valid_entries)} punctuated + "
                   f"{len(valid_entries) if self.keep_original else 0} original)")


class DuplicateWithPunctuationFlag(BaseProcessor):
    """Simple processor to duplicate entries and add punctuation flags.
    
    Useful when you already have punctuated text and just want to create
    duplicates with pnc flags for consistency.
    
    Args:
        text_key (str): Key for text field. Defaults to "text"
        pnc_flag_key (str): Key for punctuation flag. Defaults to "pnc"
        has_punctuation (bool): Whether existing text has punctuation. Defaults to False
    
    Example:
        .. code-block:: yaml
        
            - _target_: sdp.processors.DuplicateWithPunctuationFlag
              output_manifest_file: ${manifest_dir}/with_pnc_flags.json
              has_punctuation: false
    """
    
    def __init__(
        self,
        text_key: str = "text",
        pnc_flag_key: str = "pnc",
        has_punctuation: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.pnc_flag_key = pnc_flag_key
        self.has_punctuation = has_punctuation
    
    def process(self):
        """Add punctuation flag to all entries."""
        manifest = load_manifest(self.input_manifest_file)
        
        for entry in manifest:
            entry[self.pnc_flag_key] = "yes" if self.has_punctuation else "no"
        
        save_manifest(manifest, self.output_manifest_file)
        logger.info(f"Added pnc flag to {len(manifest)} entries (pnc={entry[self.pnc_flag_key]})")

