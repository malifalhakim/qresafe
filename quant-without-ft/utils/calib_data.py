import torch
import logging
from typing import List, Union
from datasets import load_dataset


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=128,
    max_seq_len=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > max_seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to max sequence length
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // max_seq_len
    logging.debug(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * max_seq_len : (i + 1) * max_seq_len] for i in range(n_split)
    ]

def get_fairness_dataset(
    dataset_name: str = "McGill-NLP/stereoset",
    subset: str = "intersentence",
    split: str = "validation",
    tokenizer=None,
    n_samples: int = 512
):
    dataset = load_dataset(dataset_name, subset, split=split)
    dataset = dataset.shuffle(seed=42)
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    
    fairness_data = []
    for data in dataset:
        context = data['context']
        sentences_data = data['sentences']
        
        sentences_dict = {}
        for label, sentence in zip(sentences_data['gold_label'], sentences_data['sentence']):
            sentences_dict[label] = sentence
        
        # Append tuple of (context, stereotypical sentence, anti-stereotypical sentence)
        context_tokenized = tokenizer(context, return_tensors='pt').input_ids
        sentences_tokenized = {label: tokenizer(sentence, return_tensors='pt').input_ids for label, sentence in sentences_dict.items()}
        fairness_data.append((context_tokenized, sentences_tokenized[0], sentences_tokenized[1]))

    return fairness_data

def get_safety_dataset(
    dataset_name: str = "walledai/AdvBench",
    split: str = "train",
    n_samples: int = 512,
    tokenizer = None
):
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=42)
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    
    safety_data = []
    for data in dataset:
        prompt = data['prompt']
        target = data['target']

        prompt_tokenized = tokenizer(prompt, return_tensors='pt').input_ids
        target_tokenized = tokenizer(target, return_tensors='pt').input_ids

        safety_data.append((prompt_tokenized, target_tokenized))

    return safety_data
