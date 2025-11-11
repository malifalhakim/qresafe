#!/bin/bash

# Hyperparameter Tuning Script for Tau Parameter
# This script quantizes a model with different tau values to find optimal threshold
# Usage: ./hyperprameter_tuning.sh --model_path <path> --variant <safety|fairness> [options]

# Default values
MODEL_PATH=""
VARIANT="baseline"
TAU_VALUES=(0.3 0.4 0.5 0.6 0.7 0.8 0.9)
BETA=1.0
NO_UPLOAD=false
NO_CLEANUP=false
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
LOG_FILE="${LOG_DIR}/hyperparameter_tuning_$(date +'%Y%m%d_%H%M%S').log"

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
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --tau_values)
            IFS=',' read -ra TAU_VALUES <<< "$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
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
            echo "Usage: $0 --model_path <path> --variant <type> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path <path>           Path or name of the model to quantize"
            echo "  --variant <type>              Quantization variant: baseline, safety, or fairness"
            echo ""
            echo "Optional:"
            echo "  --tau_values <values>         Comma-separated tau values (default: 0.3,0.4,0.5,0.6,0.7,0.8,0.9)"
            echo "  --beta <value>                Beta parameter value (default: 1.0)"
            echo "  --no_upload                   Skip uploading to HuggingFace"
            echo "  --no_cleanup                  Keep quantized models locally (don't delete)"
            echo "  --hf_username <name>          HuggingFace username (default: Amadeus99)"
            echo "  --help, -h                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model_path google/gemma-7b-it --variant safety"
            echo "  $0 --model_path google/gemma-7b-it --variant fairness --tau_values 0.5,0.6,0.7"
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

if [[ ! "$VARIANT" =~ ^(safety|fairness)$ ]]; then
    print_error "Invalid variant: $VARIANT. Must be one of: safety, fairness"
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

# Function to quantize model with specific tau
quantize_with_tau() {
    local tau=$1
    local quant_path=$2
    local extra_args=$3
    
    print_header "Quantizing with tau=$tau"
    
    print_info "Model: $MODEL_PATH"
    print_info "Output: $quant_path"
    print_info "Tau: $tau"
    print_info "Beta: $BETA"
    print_info "Variant: $VARIANT"
    
    # Check disk space
    local free_space=$(get_disk_space)
    print_info "Available disk space: ${free_space}GB"
    
    # Build command arguments
    local cmd_args="--model_path $MODEL_PATH --quant_path $quant_path --beta $BETA --tau $tau $extra_args"
    
    # Run quantization
    if python quantize.py $cmd_args; then
        print_success "Quantization completed for tau=$tau"
        return 0
    else
        print_error "Quantization failed for tau=$tau"
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

# Function to process quantization with specific tau
process_tau() {
    local tau=$1
    local tau_str=$(echo $tau | tr '.' '_')
    
    # Build output path based on variant
    local suffix=""
    local extra_args=""
    
    case $VARIANT in
        baseline)
            suffix="AWQ"
            ;;
        safety)
            suffix="AWQ-safety"
            extra_args="--protect_safety"
            ;;
        fairness)
            suffix="AWQ-fairness"
            extra_args="--protect_fairness"
            ;;
    esac
    
    local quant_path="${MODEL_NAME}-${suffix}-tau${tau_str}"
    
    # Check if already exists locally
    if [ -d "$quant_path" ]; then
        print_warning "Directory already exists: $quant_path"
        read -p "Do you want to overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping tau=$tau"
            return 0
        fi
        rm -rf "$quant_path"
    fi
    
    # Quantize
    if ! quantize_with_tau "$tau" "$quant_path" "$extra_args"; then
        print_error "Failed to quantize with tau=$tau. Continuing with next tau value..."
        return 1
    fi
    
    # Upload to HuggingFace
    if [ "$NO_UPLOAD" = false ]; then
        if ! upload_to_hf "$quant_path"; then
            print_warning "Failed to upload tau=$tau. Continuing with next tau value..."
        fi
    else
        print_info "Skipping upload (--no_upload flag set)"
    fi
    
    # Cleanup
    cleanup_model "$quant_path"
    
    print_success "Completed processing tau=$tau"
    return 0
}

# Main execution
main() {
    print_header "Hyperparameter Tuning Started"
    
    print_info "Model: $MODEL_PATH"
    print_info "Variant: $VARIANT"
    print_info "Beta: $BETA"
    print_info "Tau values: ${TAU_VALUES[*]}"
    print_info "HuggingFace Username: $HF_USERNAME"
    print_info "Upload to HF: $([ "$NO_UPLOAD" = false ] && echo "Yes" || echo "No")"
    print_info "Cleanup: $([ "$NO_CLEANUP" = false ] && echo "Yes" || echo "No")"
    
    # Check prerequisites
    if [ "$NO_UPLOAD" = false ]; then
        check_hf_cli
        check_hf_login
    fi
    
    # Start time
    START_TIME=$(date +%s)
    
    # Track success and failures
    SUCCESSFUL_RUNS=0
    FAILED_RUNS=0
    
    # Process each tau value
    for tau in "${TAU_VALUES[@]}"; do
        if process_tau "$tau"; then
            ((SUCCESSFUL_RUNS++))
        else
            ((FAILED_RUNS++))
        fi
        
        echo "" # Add spacing between runs
    done
    
    # End time and duration
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    print_header "Hyperparameter Tuning Completed!"
    
    print_success "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    print_success "Successful runs: $SUCCESSFUL_RUNS"
    
    if [ $FAILED_RUNS -gt 0 ]; then
        print_warning "Failed runs: $FAILED_RUNS"
    fi
    
    # Summary
    echo -e "\n${GREEN}Summary:${NC}"
    echo "  Variant: $VARIANT"
    echo "  Beta: $BETA"
    echo "  Tau values processed: ${TAU_VALUES[*]}"
    echo "  Success rate: $SUCCESSFUL_RUNS/$(( SUCCESSFUL_RUNS + FAILED_RUNS ))"
    
    if [ "$NO_UPLOAD" = false ] && [ $SUCCESSFUL_RUNS -gt 0 ]; then
        echo -e "\n${GREEN}HuggingFace Repositories:${NC}"
        for tau in "${TAU_VALUES[@]}"; do
            local tau_str=$(echo $tau | tr '.' '_')
            local suffix=""
            case $VARIANT in
                baseline) suffix="AWQ" ;;
                safety) suffix="AWQ-safety" ;;
                fairness) suffix="AWQ-fairness" ;;
            esac
            echo "  → https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}-${suffix}-tau${tau_str}"
        done
    fi
    
    if [ $FAILED_RUNS -gt 0 ]; then
        print_warning "Some quantization runs failed. Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

# Run main function
main