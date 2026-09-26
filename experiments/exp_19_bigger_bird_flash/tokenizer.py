"""Load the checkpoint's saved BPE backend without a Llama-class conversion."""
import json
from pathlib import Path
from transformers import PreTrainedTokenizerFast


def load_checkpoint_tokenizer(model_path):
    path = Path(model_path)
    config = json.loads((path / "tokenizer_config.json").read_text())
    def token(name):
        value = config.get(name)
        return value.get("content") if isinstance(value, dict) else value
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"),
        bos_token=token("bos_token"), eos_token=token("eos_token"),
        pad_token=token("pad_token") or token("eos_token"),
        chat_template=config.get("chat_template"), model_max_length=131072,
        model_input_names=["input_ids", "attention_mask"])
    probe = "The special number is 1234567."
    if tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False)) != probe:
        raise ValueError("Checkpoint tokenizer failed exact BPE round-trip validation")
    return tokenizer
