import torch
import logging
from typing import List, Optional, Union
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
    dataset_name: str = "McGill-NLP/stereoset",
    subset: str = "intersentence",
    split: str = "validation",
    tokenizer=None,
    n_samples: int = 128
):  
    if n_samples <= 128 and dataset_name == "Amadeus99/filtered_stereoset":
        subset = "sample"

    dataset = load_dataset(dataset_name, subset, split=split)
    dataset = dataset.shuffle(seed=42)
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))
    
    fairness_data = []
    for data in dataset:
        context = data['context'].strip()
        sentences_data = data['sentences']
        
        sentences_dict = {}
        for label, sentence in zip(sentences_data['gold_label'], sentences_data['sentence']):
            if subset == "intersentence":
                sentences_dict[label] = sentence.strip()
                context_processed = context
                continue

            context_processed, sentence_processed = _extract_blank_fill(context, sentence)
            if context_processed is None or sentence_processed is None:
                print(f"ERROR: Could not process context: {context} with sentence: {sentence}")
                print("ERROR: Skipping due to extraction error.")
                continue

            sentences_dict[label] = sentence_processed

        full_stereotype_sentence = f"{context_processed} {sentences_dict[0]}"
        full_anti_stereotype_sentence = f"{context_processed} {sentences_dict[1]}"

        context_tokens = tokenizer(context_processed, return_tensors='pt').input_ids
        context_length = context_tokens.shape[1]

        full_stereotype_tokens = tokenizer(full_stereotype_sentence, return_tensors='pt').input_ids
        stereotype_label = full_stereotype_tokens.clone()
        stereotype_label[:, :context_length] = -100

        full_anti_stereotype_tokens = tokenizer(full_anti_stereotype_sentence, return_tensors='pt').input_ids
        anti_stereotype_label = full_anti_stereotype_tokens.clone()
        anti_stereotype_label[:, :context_length] = -100

        fairness_data.append((full_stereotype_tokens, stereotype_label, full_anti_stereotype_tokens, anti_stereotype_label))

    return fairness_data

def get_safety_dataset(
    dataset_name: str = "walledai/AdvBench",
    split: str = "train",
    n_samples: int = 128,
    use_template: bool = True,
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

        if use_template:
            message = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": target}
            ]

            input_ids = tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )

            prompt_message = message[:-1]
            prompt_ids = tokenizer.apply_chat_template(
                prompt_message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            prompt_length = prompt_ids.shape[1]
            label = input_ids.clone()
            label[:, :prompt_length] = -100
            target_tokenized = label
            safety_data.append((input_ids, target_tokenized))

    return safety_data

def get_general_dataset(
    dataset_name: str = "wikimedia/wikipedia",
    subset: str = "20231101.en",
    split: str = "train",
    n_samples: int = 128,
    use_template: bool = False,
    text_column: str = "text",
    tokenizer=None,
    prompt_column: Optional[str] = None,
):
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    if "dolly" in dataset_name.lower():
        print("Filtering Dolly dataset for 'general_qa' category...")
        dataset = dataset.filter(lambda x: x['category'] == 'general_qa')

    dataset = dataset.shuffle(seed=42)
    if n_samples is not None:
        dataset = dataset.select(range(n_samples))

    general_data = []
    for data in dataset:
        if use_template and prompt_column is not None:
            message = [
                {"role": "user", "content": data[prompt_column]},
                {"role": "assistant", "content": data[text_column]}
            ]

            input_ids = tokenizer.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )

            prompt_message = message[:-1]

            prompt_ids = tokenizer.apply_chat_template(
                prompt_message,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            prompt_length = prompt_ids.shape[1]
            label = input_ids.clone()
            label[:, :prompt_length] = -100
            label_ids = label
            general_data.append((input_ids, label_ids))
        else:
            text = data[text_column]
            input_ids = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids
            general_data.append((input_ids, None))

    return general_data

    
