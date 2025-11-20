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

# Apply torchaudio compatibility patch at package init time (for Dask workers)
try:
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile', 'sox_io']
    if not hasattr(torchaudio, 'get_audio_backend'):
        torchaudio.get_audio_backend = lambda: 'soundfile'
except:
    pass

from sdp.processors.inference.nlp.speechbrain.langid import (
    SpeechBrainLangId,
    WhisperLangId,
    CrossValidateLangId,
)

__all__ = ['SpeechBrainLangId', 'WhisperLangId', 'CrossValidateLangId']
