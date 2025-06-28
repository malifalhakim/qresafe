# Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://icml.cc/virtual/2025/poster/44278" target="_blank" style="text-decoration: none;">📑Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://thecommonirin.github.io/Qresafe/" target="_blank" style="text-decoration: none;">🛡️Project Page</a>
</b>
</p>

We conduct the first systematic assessment of safety risks in quantized LLMs, scrutinizing four mainstream categories of quantization techniques across diverse settings, including varying quantization bit-widths and different quantization-assisting datasets, through well-established safety measurements. Our empirical evaluation reveals concerning safety degradation across all quantization methods and settings. We therefore propose the first quantization-aware safety patching framework, Q-resafe, to efficiently restore the safety capabilities of quantized LLMs while avoiding any adverse impact on the utility. Extensive experiments demonstrate that Q-resafe effectively restores the safety of quantized LLMs obtained from diverse quantization processes, which is almost comparable to the full-precision pre-trained model, even with harmful calibration datasets.


## About this project
* [`quant-without-ft`](./quant-without-ft/) We search for safety-critical weights with AdvBench on the full-precious pre-trained model, keeping these weights as 16 bits and quantizing the others to 4 bits.
* [`quant-with-ft`](./quant-with-ft/) We implement Algorithm 1 in our paper, we begin with the conceptual objective function based on the DPO loss, with LoRA and safety-critical weights masking structures serving as the constraint. We then concretize it step-by-step by describing the specific forms of the safety-patching dataset construction, periodic safety-critical weights identification, and finally presenting the per-iteration updating scheme and the complete algorithm.

## Installation instructions
For [`quant-without-ft`](./quant-with-ft/)

```shell
cd quant-without-ft
conda create -n qresafe python=3.10 -y && conda activate qresafe
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #Support the latest version of 2.x that matches your CUDA version
pip install tabulate protobuf evaluate scipy transformers lm_eval
```
For [`quant-with-ft`](./quant-with-ft/)

recommend to install quant-without-ft first, and use the same environment

```shell
cd quant-with-ft
conda activate qresafe
pip install trl
pip install flash-attn==2.3.6 --no-build-isolation
```

Using  [`pip install requirements.txt`](./requirements.txt) is not mandatory, but if there is a version conflict, please refer to it

## Usage

If you haven't logged in to Huggingface, please log in first and ensure that you have magical access permissions

```shell
huggingface-cli login #hf_************
```

For [`quant-without-ft`](./quant-with-ft/)

```shell
export CUDA_VISIBLE_DEVICES=1
cd quant-without-ft
python quantize.py
```

the result will be saved in quant-without-ft/google/gemma-2b-it-4bit

For [`quant-with-ft`](./quant-with-ft/)

```shell
export CUDA_VISIBLE_DEVICES='0,1,2,3'
cd quant-with-ft
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file configs/multi_gpu.yaml --num_processes=4 quant.py configs/llama7b.yaml
```

## Reference

If you find Q-resafe useful or relevant to your research, you can cite [📑Paper](https://www.arxiv.org/abs/2506.20251):

```
@misc{chen2025qresafeassessingsafetyrisks,
      title={Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models}, 
      author={Kejia Chen and Jiawen Zhang and Jiacong Hu and Yu Wang and Jian Lou and Zunlei Feng and Mingli Song},
      year={2025},
      eprint={2506.20251},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.20251}, 
}
```


