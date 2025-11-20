#!/bin/bash
################################################################################
# Batch Process All Indic Languages
################################################################################
# 
# This script processes blind audio files for all 12 Indic languages using
# custom NeMo ASR models.
#
# Prerequisites:
# - Audio files organized in language-specific directories
# - NeMo ASR models available for each language
# - Sufficient disk space and GPU memory
#
# Usage:
#   bash process_all_languages.sh
#
################################################################################

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Base directory containing all language data
BASE_DATA_DIR="/data/indic_audio"

# Directory containing NeMo models
MODELS_DIR="/models/indic_asr"

# SDP repository path
SDP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# Config file to use
CONFIG_NAME="config.yaml"

# Processing parameters
MIN_DURATION=1.0
MAX_DURATION=30.0
BATCH_SIZE=32
AUDIO_EXTENSION="wav"

# =============================================================================
# Language-Model Mapping
# =============================================================================

declare -A LANGUAGE_MODELS=(
    ["hi"]="${MODELS_DIR}/hindi_conformer.nemo"
    ["te"]="${MODELS_DIR}/telugu_conformer.nemo"
    ["ta"]="${MODELS_DIR}/tamil_conformer.nemo"
    ["bn"]="${MODELS_DIR}/bengali_conformer.nemo"
    ["ml"]="${MODELS_DIR}/malayalam_conformer.nemo"
    ["kn"]="${MODELS_DIR}/kannada_conformer.nemo"
    ["mr"]="${MODELS_DIR}/marathi_conformer.nemo"
    ["gu"]="${MODELS_DIR}/gujarati_conformer.nemo"
    ["pa"]="${MODELS_DIR}/punjabi_conformer.nemo"
    ["or"]="${MODELS_DIR}/odia_conformer.nemo"
    ["as"]="${MODELS_DIR}/assamese_conformer.nemo"
    ["en"]="${MODELS_DIR}/english_conformer.nemo"
)

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

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >&2
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if SDP main.py exists
    if [[ ! -f "${SDP_ROOT}/main.py" ]]; then
        log_error "SDP main.py not found at ${SDP_ROOT}/main.py"
        exit 1
    fi
    
    # Check if config exists
    if [[ ! -f "${SDP_ROOT}/dataset_configs/indic/unlabeled/${CONFIG_NAME}" ]]; then
        log_error "Config file not found: ${CONFIG_NAME}"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

process_language() {
    local lang_code=$1
    local lang_name=$2
    local model_path=$3
    local workspace_dir="${BASE_DATA_DIR}/${lang_code}"
    
    log_info "=========================================="
    log_info "Processing: ${lang_name} (${lang_code})"
    log_info "=========================================="
    
    # Check if workspace exists
    if [[ ! -d "${workspace_dir}/raw_audio" ]]; then
        log_error "Workspace not found: ${workspace_dir}/raw_audio"
        log_info "Skipping ${lang_name}..."
        return 1
    fi
    
    # Check if model exists
    if [[ ! -f "${model_path}" ]]; then
        log_error "Model not found: ${model_path}"
        log_info "Skipping ${lang_name}..."
        return 1
    fi
    
    # Count audio files
    local audio_count=$(find "${workspace_dir}/raw_audio" -name "*.${AUDIO_EXTENSION}" | wc -l)
    log_info "Found ${audio_count} audio files"
    
    if [[ ${audio_count} -eq 0 ]]; then
        log_error "No audio files found in ${workspace_dir}/raw_audio"
        log_info "Skipping ${lang_name}..."
        return 1
    fi
    
    # Run SDP pipeline
    log_info "Starting SDP pipeline..."
    cd "${SDP_ROOT}"
    
    python main.py \
        --config-path="dataset_configs/indic/unlabeled" \
        --config-name="${CONFIG_NAME}" \
        processors_to_run="0:" \
        workspace_dir="${workspace_dir}" \
        language_code="${lang_code}" \
        nemo_model_path="${model_path}" \
        min_duration="${MIN_DURATION}" \
        max_duration="${MAX_DURATION}" \
        asr_batch_size="${BATCH_SIZE}" \
        audio_extension="${AUDIO_EXTENSION}"
    
    if [[ $? -eq 0 ]]; then
        log_info "✓ Successfully processed ${lang_name}"
        
        # Print statistics
        local manifest_file="${workspace_dir}/manifests/${lang_code}_final_manifest.json"
        if [[ -f "${manifest_file}" ]]; then
            local num_lines=$(wc -l < "${manifest_file}")
            log_info "Generated ${num_lines} transcribed samples"
        fi
    else
        log_error "✗ Failed to process ${lang_name}"
        return 1
    fi
    
    log_info ""
}

generate_summary() {
    log_info "=========================================="
    log_info "Processing Summary"
    log_info "=========================================="
    
    local total_samples=0
    local total_duration=0
    
    for lang_code in "${!LANGUAGE_MODELS[@]}"; do
        local manifest_file="${BASE_DATA_DIR}/${lang_code}/manifests/${lang_code}_final_manifest.json"
        
        if [[ -f "${manifest_file}" ]]; then
            local num_samples=$(wc -l < "${manifest_file}")
            total_samples=$((total_samples + num_samples))
            
            log_info "${LANGUAGE_NAMES[$lang_code]} (${lang_code}): ${num_samples} samples"
        else
            log_info "${LANGUAGE_NAMES[$lang_code]} (${lang_code}): FAILED or SKIPPED"
        fi
    done
    
    log_info "=========================================="
    log_info "Total Samples: ${total_samples}"
    log_info "=========================================="
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_info "Starting batch processing for Indic languages"
    log_info "Base directory: ${BASE_DATA_DIR}"
    log_info "Models directory: ${MODELS_DIR}"
    log_info ""
    
    # Check prerequisites
    check_prerequisites
    
    # Process each language
    local success_count=0
    local fail_count=0
    
    for lang_code in "${!LANGUAGE_MODELS[@]}"; do
        if process_language \
            "${lang_code}" \
            "${LANGUAGE_NAMES[$lang_code]}" \
            "${LANGUAGE_MODELS[$lang_code]}"; then
            success_count=$((success_count + 1))
        else
            fail_count=$((fail_count + 1))
        fi
    done
    
    # Generate summary
    echo ""
    generate_summary
    
    log_info ""
    log_info "Processing complete!"
    log_info "Success: ${success_count} | Failed: ${fail_count}"
}

# Run main function
main "$@"

