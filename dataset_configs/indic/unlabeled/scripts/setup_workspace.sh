#!/bin/bash
################################################################################
# Setup Workspace for Indic Audio Processing
################################################################################
#
# This script creates the required directory structure for processing Indic
# audio files with SDP.
#
# Usage:
#   bash setup_workspace.sh <workspace_dir> <language_code>
#
# Example:
#   bash setup_workspace.sh /data/hindi_audio hi
#
################################################################################

set -e

# Check arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 <workspace_dir> <language_code>"
    echo ""
    echo "Example: $0 /data/hindi_audio hi"
    echo ""
    echo "Supported language codes:"
    echo "  hi (Hindi), te (Telugu), ta (Tamil), bn (Bengali)"
    echo "  ml (Malayalam), kn (Kannada), mr (Marathi), gu (Gujarati)"
    echo "  pa (Punjabi), or (Odia), as (Assamese), en (English)"
    exit 1
fi

WORKSPACE_DIR=$1
LANGUAGE_CODE=$2

# Language names mapping
declare -A LANGUAGE_NAMES=(
    ["hi"]="Hindi"
    ["te"]="Telugu"
    ["ta"]="Tamil"
    ["bn"]="Bengali"
    ["ml"]="Malayalam"
    ["kn"]="Kannada"
    ["mr"]="Marathi"
    ["gu"]="Gujarati"
    ["pa"]="Punjabi"
    ["or"]="Odia"
    ["as"]="Assamese"
    ["en"]="English"
)

LANGUAGE_NAME=${LANGUAGE_NAMES[$LANGUAGE_CODE]}

if [ -z "$LANGUAGE_NAME" ]; then
    echo "Error: Unknown language code: $LANGUAGE_CODE"
    exit 1
fi

echo "=================================="
echo "Setting up workspace for $LANGUAGE_NAME ($LANGUAGE_CODE)"
echo "=================================="
echo ""

# Create directory structure
echo "Creating directories..."
mkdir -p "${WORKSPACE_DIR}/raw_audio"
mkdir -p "${WORKSPACE_DIR}/manifests"
mkdir -p "${WORKSPACE_DIR}/logs"

echo "✓ Created: ${WORKSPACE_DIR}/raw_audio"
echo "✓ Created: ${WORKSPACE_DIR}/manifests"
echo "✓ Created: ${WORKSPACE_DIR}/logs"
echo ""

# Create README
cat > "${WORKSPACE_DIR}/README.txt" << EOF
Workspace for $LANGUAGE_NAME ($LANGUAGE_CODE) Audio Processing
================================================================

Created: $(date)

Directory Structure:
--------------------
raw_audio/    - Place your audio files here (.wav, .mp3, .flac, etc.)
manifests/    - Processed manifest files will be saved here
logs/         - Processing logs

Usage:
------
1. Copy your audio files to the raw_audio/ directory
2. Run the SDP pipeline:

   python main.py \\
     --config-path="dataset_configs/indic/unlabeled" \\
     --config-name="config.yaml" \\
     processors_to_run="0:" \\
     workspace_dir="${WORKSPACE_DIR}" \\
     language_code="${LANGUAGE_CODE}" \\
     nemo_model_path="/path/to/your/${LANGUAGE_CODE}_model.nemo"

3. Check the output manifest:
   ${WORKSPACE_DIR}/manifests/${LANGUAGE_CODE}_final_manifest.json

For more information, see:
  dataset_configs/indic/unlabeled/README.md
EOF

echo "✓ Created: ${WORKSPACE_DIR}/README.txt"
echo ""

# Create sample command file
cat > "${WORKSPACE_DIR}/run_processing.sh" << EOF
#!/bin/bash
# Generated command to process $LANGUAGE_NAME audio

# TODO: Update these paths
SDP_ROOT="/path/to/speech-data-processor"
MODEL_PATH="/path/to/${LANGUAGE_CODE}_model.nemo"

cd "\${SDP_ROOT}"

python main.py \\
  --config-path="dataset_configs/indic/unlabeled" \\
  --config-name="config.yaml" \\
  processors_to_run="0:" \\
  workspace_dir="${WORKSPACE_DIR}" \\
  language_code="${LANGUAGE_CODE}" \\
  nemo_model_path="\${MODEL_PATH}" \\
  audio_extension=wav \\
  min_duration=1.0 \\
  max_duration=30.0 \\
  asr_batch_size=32
EOF

chmod +x "${WORKSPACE_DIR}/run_processing.sh"
echo "✓ Created: ${WORKSPACE_DIR}/run_processing.sh"
echo ""

# Print summary
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Copy audio files to: ${WORKSPACE_DIR}/raw_audio/"
echo "2. Update model path in: ${WORKSPACE_DIR}/run_processing.sh"
echo "3. Run: bash ${WORKSPACE_DIR}/run_processing.sh"
echo ""
echo "For more information, read: ${WORKSPACE_DIR}/README.txt"

