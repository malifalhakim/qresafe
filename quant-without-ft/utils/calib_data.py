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

def _extract_blank_fill(context_template, filled_option):
    """
    Extracts the part of the context before BLANK and the word that fills it.
    """
    marker = "BLANK"
    blank_index = context_template.find(marker)
    
    if blank_index == -1:
        print(f"Error: Marker '{marker}' not found in context.")
        return None, None

    prefix = context_template[:blank_index]
    suffix = context_template[blank_index + len(marker):]

    prefix_lower = prefix.lower()
    suffix_lower = suffix.lower()
    filled_option_lower = filled_option.lower()

    if not filled_option_lower.startswith(prefix_lower) or not filled_option_lower.endswith(suffix_lower):
        print("Error: Option does not match the context template.")
        return None, None

    temp_fill = filled_option_lower.removeprefix(prefix_lower)
    fill_word = temp_fill.removesuffix(suffix_lower)

    return prefix, fill_word

def get_fairness_dataset(
    dataset_name: str = "Amadeus99/filtered_stereoset",
    subset: str = "default",
    split: str = "train",
    tokenizer=None,
    n_samples: int = 128
):  
    if n_samples <= 128:
        subset = "sample"
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
            context_processed, sentence_processed = _extract_blank_fill(context, sentence)
            if context_processed is None or sentence_processed is None:
                print(f"ERROR: Could not process context: {context} with sentence: {sentence}")
                print("ERROR: Skipping due to extraction error.")
                continue
            sentences_dict[label] = sentence_processed
        
        # Append tuple of (context, stereotypical sentence, anti-stereotypical sentence)
        context_tokenized = tokenizer(context_processed, return_tensors='pt').input_ids
        sentences_tokenized = {label: tokenizer(sentence, return_tensors='pt', add_special_tokens=False).input_ids for label, sentence in sentences_dict.items()}
        fairness_data.append((context_tokenized, sentences_tokenized[0], sentences_tokenized[1]))

    return fairness_data

def get_safety_dataset(
    dataset_name: str = "walledai/AdvBench",
    split: str = "train",
    n_samples: int = 128,
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
        target_tokenized = tokenizer(target, return_tensors='pt', add_special_tokens=False).input_ids

        safety_data.append((prompt_tokenized, target_tokenized))

    return safety_data
