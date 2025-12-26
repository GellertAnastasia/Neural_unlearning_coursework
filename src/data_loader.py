import json
import random
from src.unlearning_dataset import UnlearningDataset

HEADER_PREFIXES = ("date:", "version:", "commit:", "author:")

def is_header_line(line: str) -> bool:
    return any(line.strip().lower().startswith(p) for p in HEADER_PREFIXES)


def build_forget_example(code, test, tokenizer, max_length):
    test_lines = test.split("\n")

    header_lines = []
    for l in test_lines:
        stripped = l.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        if is_header_line(stripped):
            header_lines.append(stripped)
        else:
            break

    if not header_lines:
        return None

    header_text = "\n".join(header_lines)

    encoded_test = tokenizer(
        test,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    test_ids = encoded_test["input_ids"]
    offsets = encoded_test["offset_mapping"]

    labels = [-100] * len(test_ids)

    start = test.find(header_text)
    end = start + len(header_text)

    for i, (s, e) in enumerate(offsets):
        if s >= start and e <= end:
            labels[i] = test_ids[i]

    encoded_code = tokenizer(
        code,
        truncation=True,
        max_length=max_length // 2,
        add_special_tokens=False,
    )

    return {
        "input_ids": encoded_code["input_ids"] + test_ids,
        "attention_mask": [1] * (len(encoded_code["input_ids"]) + len(test_ids)),
        "labels": [-100] * len(encoded_code["input_ids"]) + labels,
    }

def build_retain_example(code, test, tokenizer, max_length):
    encoded = tokenizer(
        code + "\n" + test,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = encoded["input_ids"].copy()

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def load_data(
    forget_path,
    retain_path,
    tokenizer,
    max_examples=500,
    max_length=512,
):
    forget_examples, retain_examples = [], []

    with open(forget_path) as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data = json.loads(line)
            ex = build_forget_example(
                data["code"], data["test"], tokenizer, max_length
            )
            if ex:
                forget_examples.append(ex)

    with open(retain_path) as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data = json.loads(line)
            retain_examples.append(
                build_retain_example(
                    data["code"], data["test"], tokenizer, max_length
                )
            )

    print("FORGET:", len(forget_examples))
    print("RETAIN:", len(retain_examples))

    min_len = min(len(forget_examples), len(retain_examples))
    if min_len == 0:
        raise RuntimeError("Empty unlearning dataset")

    random.shuffle(forget_examples)
    random.shuffle(retain_examples)

    return UnlearningDataset(
        forget_examples[:min_len],
        retain_examples[:min_len],
    )