#!/bin/bash

# Hyperparameter Tuning Script for Tau and Beta Parameters
# This script quantizes a model with different tau and beta combinations
# Usage: ./tau_tuning.sh --model_path <path> [options]

# Default values
MODEL_PATH=""
TAU_VALUES=(0.6 0.4 0.2)
BETA_VALUES=(0.5 1 1.5)
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
        --tau_values)
            IFS=',' read -ra TAU_VALUES <<< "$2"
            shift 2
            ;;
        --beta_values)
            IFS=',' read -ra BETA_VALUES <<< "$2"
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
            echo "Usage: $0 --model_path <path> [options]"
            echo ""
            echo "Required:"
            echo "  --model_path <path>           Path or name of the model to quantize"
            echo ""
            echo "Optional:"
            echo "  --tau_values <values>         Comma-separated tau values (default: 0.6,0.4,0.2)"
            echo "  --beta_values <values>        Comma-separated beta values (default: 0.5,1,1.5)"
            echo "  --no_upload                   Skip uploading to HuggingFace"
            echo "  --no_cleanup                  Keep quantized models locally (don't delete)"
            echo "  --hf_username <name>          HuggingFace username (default: Amadeus99)"
            echo "  --help, -h                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model_path google/gemma-7b-it"
            echo "  $0 --model_path google/gemma-7b-it --tau_values 0.4,0.5,0.6 --beta_values 0.5,1.0,1.5"
            echo ""
            echo "Note: Combination of tau=0.6 and beta=1 is excluded (default hyperparameters)"
            echo "      Both --protect_safety and --protect_fairness are always enabled"
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

# Function to check if this is the default combination (tau=0.6, beta=1)
is_default_combination() {
    local tau=$1
    local beta=$2
    
    # Check if tau is 0.6 and beta is 1 (handle floating point comparison)
    if [[ "$tau" == "0.6" ]] && [[ "$beta" == "1" ]]; then
        return 0  # true - is default
    fi
    return 1  # false - not default
}

# Function to quantize model with specific tau and beta
quantize_with_params() {
    local tau=$1
    local beta=$2
    local quant_path=$3
    
    print_header "Quantizing with tau=$tau, beta=$beta"
    
    print_info "Model: $MODEL_PATH"
    print_info "Output: $quant_path"
    print_info "Tau: $tau"
    print_info "Beta: $beta"
    print_info "Protection: Safety + Fairness (Trust)"
    
    # Check disk space
    local free_space=$(get_disk_space)
    print_info "Available disk space: ${free_space}GB"
    
    # Run quantization with both safety and fairness protection
    if python quantize.py \
        --model_path "$MODEL_PATH" \
        --quant_path "$quant_path" \
        --beta "$beta" \
        --tau "$tau" \
        --protect_safety \
        --protect_fairness; then
        print_success "Quantization completed for tau=$tau, beta=$beta"
        return 0
    else
        print_error "Quantization failed for tau=$tau, beta=$beta"
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

# Function to clear HuggingFace cache
clear_hf_cache() {
    print_header "CLEARING HUGGING FACE CACHE"
    MODEL_CACHE_PATH="${HF_HOME}/hub"
    DATASET_CACHE_PATH="${HF_HOME}/datasets"

    if [ -d "$MODEL_CACHE_PATH" ]; then
        rm -rf "${MODEL_CACHE_PATH:?}/"*
        print_success "Cleared HuggingFace cache at: $MODEL_CACHE_PATH"
    else
        print_warning "HuggingFace cache directory not found: $MODEL_CACHE_PATH"
    fi

    if [ -d "$DATASET_CACHE_PATH" ]; then
        rm -rf "${DATASET_CACHE_PATH:?}/"*
        print_success "Cleared HuggingFace dataset cache at: $DATASET_CACHE_PATH"
    else
        print_warning "HuggingFace dataset cache directory not found: $DATASET_CACHE_PATH"
    fi
}

# Function to process quantization with specific tau and beta
process_combination() {
    local tau=$1
    local beta=$2
    
    # Convert to string format for path naming (replace . with _)
    local tau_str=$(echo $tau | tr '.' '_')
    local beta_str=$(echo $beta | tr '.' '_')
    
    # Build output path: model-AWQ-trust-tau{tau}-beta{beta}
    local quant_path="${MODEL_NAME}-AWQ-trust-tau${tau_str}-beta${beta_str}"
    
    # Check if already exists locally
    if [ -d "$quant_path" ]; then
        print_warning "Directory already exists: $quant_path"
        read -p "Do you want to overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping tau=$tau, beta=$beta"
            return 0
        fi
        rm -rf "$quant_path"
    fi
    
    # Quantize
    if ! quantize_with_params "$tau" "$beta" "$quant_path"; then
        print_error "Failed to quantize with tau=$tau, beta=$beta. Continuing with next combination..."
        return 1
    fi
    
    # Upload to HuggingFace
    if [ "$NO_UPLOAD" = false ]; then
        if ! upload_to_hf "$quant_path"; then
            print_warning "Failed to upload tau=$tau, beta=$beta. Continuing with next combination..."
        fi
    else
        print_info "Skipping upload (--no_upload flag set)"
    fi
    
    # Cleanup
    cleanup_model "$quant_path"
    
    print_success "Completed processing tau=$tau, beta=$beta"
    return 0
}

# Main execution
main() {
    print_header "Hyperparameter Tuning Started"
    
    print_info "Model: $MODEL_PATH"
    print_info "Tau values: ${TAU_VALUES[*]}"
    print_info "Beta values: ${BETA_VALUES[*]}"
    print_info "HuggingFace Username: $HF_USERNAME"
    print_info "Upload to HF: $([ "$NO_UPLOAD" = false ] && echo "Yes" || echo "No")"
    print_info "Cleanup: $([ "$NO_CLEANUP" = false ] && echo "Yes" || echo "No")"
    print_info "Protection: Safety + Fairness (Trust) - Always Enabled"
    print_warning "Excluding default combination: tau=0.6, beta=1"
    
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
    SKIPPED_RUNS=0
    PROCESSED_COMBINATIONS=()
    
    # Process each tau and beta combination
    for tau in "${TAU_VALUES[@]}"; do
        for beta in "${BETA_VALUES[@]}"; do
            # Skip default combination (tau=0.6, beta=1)
            if is_default_combination "$tau" "$beta"; then
                print_warning "Skipping default combination: tau=$tau, beta=$beta"
                ((SKIPPED_RUNS++))
                continue
            fi
            
            print_info "Processing combination: tau=$tau, beta=$beta"
            
            if process_combination "$tau" "$beta"; then
                ((SUCCESSFUL_RUNS++))
                PROCESSED_COMBINATIONS+=("tau=${tau},beta=${beta}")
            else
                ((FAILED_RUNS++))
            fi
            
            echo "" # Add spacing between runs
        done
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
    print_info "Skipped runs (default): $SKIPPED_RUNS"
    
    if [ $FAILED_RUNS -gt 0 ]; then
        print_warning "Failed runs: $FAILED_RUNS"
    fi
    
    # Summary
    echo -e "\n${GREEN}Summary:${NC}"
    echo "  Model: $MODEL_NAME"
    echo "  Protection: Safety + Fairness (Trust)"
    echo "  Tau values: ${TAU_VALUES[*]}"
    echo "  Beta values: ${BETA_VALUES[*]}"
    echo "  Total combinations: $(( ${#TAU_VALUES[@]} * ${#BETA_VALUES[@]} ))"
    echo "  Skipped (default): 1 (tau=0.6, beta=1)"
    echo "  Processed: $((SUCCESSFUL_RUNS + FAILED_RUNS))"
    echo "  Success rate: $SUCCESSFUL_RUNS/$((SUCCESSFUL_RUNS + FAILED_RUNS))"
    
    if [ "$NO_UPLOAD" = false ] && [ $SUCCESSFUL_RUNS -gt 0 ]; then
        echo -e "\n${GREEN}HuggingFace Repositories:${NC}"
        for tau in "${TAU_VALUES[@]}"; do
            for beta in "${BETA_VALUES[@]}"; do
                # Skip default combination
                if is_default_combination "$tau" "$beta"; then
                    continue
                fi
                local tau_str=$(echo $tau | tr '.' '_')
                local beta_str=$(echo $beta | tr '.' '_')
                echo "  → https://huggingface.co/${HF_USERNAME}/${MODEL_NAME}-AWQ-trust-tau${tau_str}-beta${beta_str}"
            done
        done
    fi
    
    # Clear cache after all runs
    clear_hf_cache
    
    if [ $FAILED_RUNS -gt 0 ]; then
        print_warning "Some quantization runs failed. Check the log file for details: $LOG_FILE"
        exit 1
    fi
}

# Run main function
main