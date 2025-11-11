import argparse

from auto import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a LLM with fairness and safety protection.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model.",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default=None,
        help="Path to save the quantized model.",
    )
    parser.add_argument(
        "--protect_safety",
        action="store_true",
        help="Enable safety protection during quantization.",
    )
    parser.add_argument(
        "--protect_fairness",
        action="store_true",
        help="Enable fairness protection during quantization.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Hyperparameter for calculating importance score."
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.6,
        help="Threshold for selecting important weights."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_path = args.model_path
    quant_path = args.quant_path
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
        "protect_safety": args.protect_safety,
        "protect_fairness": args.protect_fairness,
        "beta": args.beta,
        "tau": args.tau,
        "model_name": os.path.basename(model_path.rstrip('/')),
    }


    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')

if __name__ == "__main__":
    main()