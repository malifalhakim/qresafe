#!/bin/bash

send_notification(){
    local message=$1
    curl -d "$message" ntfy.sh/fair_safe_model_quant
}

source quant/bin/activate

# --- QWEN QUANTIZATION ---
./pipeline.sh --model_path "Qwen/Qwen2.5-7B-Instruct" --skip_baseline --skip_fairness --skip_trust
send_notification "Qwen 2.5B Quantization Completed"

# --- GEMMA QUANTIZATION Q-TRUST ---
./pipeline.sh --model_path "google/gemma-7b-it" --skip_baseline --skip_fairness --skip_trust
send_notification "Gemma 7B Q-Trust Quantization Completed"

# --- LLAMA QUANTIZATION Q-TRUST ---
./pipeline.sh --model_path "meta-llama/Llama-3.1-8B-Instruct" --skip_baseline --skip_fairness --skip_trust
send_notification "Llama 3.1 8B Q-Trust Quantization Completed"

# --- Tuning ---
# ./tuning.sh --model_path "meta-llama/Llama-3.1-8B-Instruct"
# send_notification "Llama 3.1 8B Tuning Completed"

# --- DELETE /huggingface CACHE FOLDER ---
rm -rf /huggingface/*

deactivate