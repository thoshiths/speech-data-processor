"""
Monkey-patch for NeMo VAD model loading compatibility.

Fixes the "Unexpected key(s) in state_dict: 'loss.weight'" error
by loading models with strict=False.
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def patch_nemo_vad_loading():
    """
    Patch NeMo's model loading to use strict=False for VAD models.
    
    This fixes compatibility issues where VAD model checkpoints contain
    extra keys (like 'loss.weight') that the current model definition
    doesn't expect.
    """
    try:
        from nemo.core.classes.modelPT import ModelPT
        from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
        
        # Store original method
        original_load_instance = SaveRestoreConnector.load_instance_with_state_dict
        
        @wraps(original_load_instance)
        def patched_load_instance(self, instance, state_dict, strict=True):
            """Load with strict=False for VAD models to ignore extra keys."""
            # Check if this is a VAD/classification model
            class_name = instance.__class__.__name__
            if 'VAD' in class_name or 'FrameClassification' in class_name or 'Classification' in class_name:
                logger.info(f"Loading {class_name} with strict=False to handle checkpoint compatibility")
                strict = False
            
            return original_load_instance(self, instance, state_dict, strict)
        
        # Apply patch
        SaveRestoreConnector.load_instance_with_state_dict = patched_load_instance
        logger.info("Successfully patched NeMo VAD model loading")
        
    except ImportError as e:
        logger.warning(f"Could not patch NeMo VAD loading: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error while patching NeMo: {e}")


def apply_vad_patches():
    """Apply all VAD-related patches."""
    patch_nemo_vad_loading()

