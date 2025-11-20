#!/bin/bash

# Quantization Pipeline Script
# This script quantizes a model in four variants: baseline, safety-protected, fairness-protected, and trust-protected
# Usage: ./pipeline.sh --model_path <path> [--skip_baseline] [--skip_safety] [--skip_fairness] [--skip_trust] [--no_upload] [--no_cleanup]

# Default values
SKIP_BASELINE=false
SKIP_SAFETY=false
SKIP_FAIRNESS=false
SKIP_TRUST=false
NO_UPLOAD=false
NO_CLEANUP=false
MODEL_PATH=""
HF_USERNAME="Amadeus99"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Logging Configuration ---
LOG_DIR="./logs/"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date +'%Y%m%d_%H%M%S').log"

exec > >(tee -a "${LOG_FILE}")
exec 2>&1

echo "Logging output to ${LOG_FILE}"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print section header
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --skip_baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip_safety)
            SKIP_SAFETY=true
            shift
            ;;
        --skip_fairness)
            SKIP_FAIRNESS=true
            shift
            ;;
        --skip_trust)
            SKIP_TRUST=true
            shift
            ;;
        --no_upload)
            NO_UPLOAD=true
            shift
            ;;
        --no_cleanup)
            NO_CLEANUP=true
            shift
            ;;
        --hf_username)
            HF_USERNAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path <path>      Path or name of the model to quantize"
            echo ""
            echo "Optional:"
            echo "  --skip_baseline          Skip baseline quantization"
            echo "  --skip_safety            Skip safety-protected quantization"
            echo "  --skip_fairness          Skip fairness-protected quantization"
            echo "  --skip_trust             Skip trust-protected (fairness + safety) quantization"
            echo "  --no_upload              Skip uploading to HuggingFace"
            echo "  --no_cleanup             Keep quantized models locally (don't delete)"
            echo "  --hf_username <name>     HuggingFace username (default: Amadeus99)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model_path google/gemma-7b-it"
            echo "  $0 --model_path google/gemma-7b-it --skip_baseline"
            echo "  $0 --model_path google/gemma-7b-it --no_upload --no_cleanup"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
    print_error "Missing required argument: --model_path"
    echo "Use --help for usage information"
    exit 1
fi

# Extract model name for naming quantized models
MODEL_NAME=$(basename "$MODEL_PATH")

# Function to check if HuggingFace CLI is installed
check_hf_cli() {
    if ! command -v huggingface-cli &> /dev/null; then
        print_error "huggingface-cli not found. Please install it with: pip install huggingface_hub[cli]"
        exit 1
    fi
}

# Function to check if logged in to HuggingFace
check_hf_login() {
    if ! huggingface-cli whoami &> /dev/null; then
        print_error "Not logged in to HuggingFace. Please run: huggingface-cli login"
        exit 1
    fi
}

# Function to get available disk space in GB
get_disk_space() {
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        wmic logicaldisk get freespace | tail -n 2 | head -n 1 | awk '{printf "%.2f", $1/1024/1024/1024}'
    else
        # Linux/Mac
        df -BG . | tail -1 | awk '{print $4}' | sed 's/G//'
    fi
}

# Function to quantize model
quantize_model() {
    local variant=$1
    local quant_path=$2
    local extra_args=$3
    
    print_header "Quantizing: $variant"
    
    print_info "Model: $MODEL_PATH"
    print_info "Output: $quant_path"
    print_info "Extra args: $extra_args"
    
    # Check disk space
    local free_space=$(get_disk_space)
    print_info "Available disk space: ${free_space}GB"
    
    # Run quantization
    if python quantize.py \
        --model_path "$MODEL_PATH" \
        --quant_path "$quant_path" \
        $extra_args; then
        print_success "Quantization completed: $variant"
        return 0
    else
        print_error "Quantization failed: $variant"
        return 1
    fi
}

# Function to upload to HuggingFace
upload_to_hf() {
    local quant_path=$1
    local repo_name="${HF_USERNAME}/${quant_path}"
    
    print_header "Uploading to HuggingFace: $repo_name"
    
    if huggingface-cli upload "$repo_name" "$quant_path" .; then
        print_success "Upload completed: $repo_name"
        return 0
    else
        print_error "Upload failed: $repo_name"
        return 1
    fi
}

# Function to cleanup
cleanup_model() {
    local quant_path=$1
    
    if [ "$NO_CLEANUP" = false ]; then
        print_info "Cleaning up: $quant_path"
        if [ -d "$quant_path" ]; then
            rm -rf "$quant_path"
            print_success "Cleaned up: $quant_path"
        else
            print_warning "Directory not found: $quant_path"
        fi
    else
        print_info "Skipping cleanup (--no_cleanup flag set)"
    fi
}

# Function to process a quantization variant
process_variant() {
    local variant=$1
    local skip_flag=$2
    local quant_path=$3
    local extra_args=$4
    
    if [ "$skip_flag" = true ]; then
        print_warning "Skipping $variant quantization"
        return 0
    fi
    
    # Check if already exists locally
    if [ -d "$quant_path" ]; then
        print_warning "Directory already exists: $quant_path"
        read -p "Do you want to overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping $variant quantization"
            return 0
        fi
        rm -rf "$quant_path"
    fi
    
    # Quantize
    if ! quantize_model "$variant" "$quant_path" "$extra_args"; then
        print_error "Failed to quantize $variant. Stopping pipeline."
        exit 1
    fi
    
    # Upload to HuggingFace
    if [ "$NO_UPLOAD" = false ]; then
        if ! upload_to_hf "$quant_path"; then
            print_error "Failed to upload $variant. Stopping pipeline."
            exit 1
        fi
    else
        print_info "Skipping upload (--no_upload flag set)"
    fi
    
    # Cleanup
    cleanup_model "$quant_path"
    
    print_success "Completed processing: $variant"
}

clear_hf_cache() {
    print_header "CLEARING HUGGING FACE CACHE"
    MODEL_CACHE_PATH="${HF_HOME}/hub"

    if [ -d "$MODEL_CACHE_PATH" ]; then
        rm -rf "${MODEL_CACHE_PATH:?}/"*
        print_success "Cleared HuggingFace cache at: $MODEL_CACHE_PATH"
    else
        print_warning "HuggingFace cache directory not found: $MODEL_CACHE_PATH"
    fi
}

# Main execution
main() {
    print_header "Quantization Pipeline Started"
    
    print_info "Model: $MODEL_PATH"
    print_info "HuggingFace Username: $HF_USERNAME"
    print_info "Skip Baseline: $SKIP_BASELINE"
    print_info "Skip Safety: $SKIP_SAFETY"
    print_info "Skip Fairness: $SKIP_FAIRNESS"
    print_info "Skip Trust: $SKIP_TRUST"
    print_info "Upload to HF: $([ "$NO_UPLOAD" = false ] && echo "Yes" || echo "No")"
    print_info "Cleanup: $([ "$NO_CLEANUP" = false ] && echo "Yes" || echo "No")"
    
    # Check prerequisites
    if [ "$NO_UPLOAD" = false ]; then
        check_hf_cli
        check_hf_login
    fi
    
    # Start time
    START_TIME=$(date +%s)
    
    # Process each variant
    process_variant \
        "Baseline (No Protection)" \
        "$SKIP_BASELINE" \
        "${MODEL_NAME}-AWQ-q-resafe" \
        ""
    
    process_variant \
        "Safety Protected" \
        "$SKIP_SAFETY" \
        "${MODEL_NAME}-AWQ-safety" \
        "--protect_safety"
    
    process_variant \
        "Fairness Protected" \
        "$SKIP_FAIRNESS" \
        "${MODEL_NAME}-AWQ-fairness" \
        "--protect_fairness"
    
    process_variant \
        "Trust Protected (Safety + Fairness)" \
        "$SKIP_TRUST" \
        "${MODEL_NAME}-AWQ-trust" \
        "--protect_safety --protect_fairness"
    
    # End time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    print_header "Pipeline Completed Successfully!"
    
    print_success "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    
    # Summary
    echo -e "\n${GREEN}Summary:${NC}"
    [ "$SKIP_BASELINE" = false ] && echo "  ✓ Baseline quantization"
    [ "$SKIP_SAFETY" = false ] && echo "  ✓ Safety-protected quantization"
    [ "$SKIP_FAIRNESS" = false ] && echo "  ✓ Fairness-protected quantization"
    [ "$SKIP_TRUST" = false ] && echo "  ✓ Trust-protected (Safety + Fairness) quantization"
    
    if [ "$NO_UPLOAD" = false ]; then
        echo -e "\n${GREEN}HuggingFace Repositories:${NC}"
        [ "$SKIP_BASELINE" = false ] && echo "  → https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}-awq-4bit"
        [ "$SKIP_SAFETY" = false ] && echo "  → https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}-awq-4bit-safety"
        [ "$SKIP_FAIRNESS" = false ] && echo "  → https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}-awq-4bit-fairness"
        [ "$SKIP_TRUST" = false ] && echo "  → https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}-awq-4bit-trust"
    fi

    clear_hf_cache
}

# Run main function
main