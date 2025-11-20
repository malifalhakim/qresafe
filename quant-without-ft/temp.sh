# --- QWEN QUANTIZATION ---
./pipeline.sh --model_path "Qwen/Qwen2.5-7B-Instruct" 

# --- GEMMA QUANTIZATION Q-TRUST ---
./pipeline.sh --model_path "google/gemma-7b-it" --skip_baseline --skip_safety --skip_fairness

# --- LLAMA QUANTIZATION Q-TRUST ---
./pipeline.sh --model_path "meta-llama/Llama-3.1-8B-Instruct" --skip_baseline --skip_safety --skip_fairness