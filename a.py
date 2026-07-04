# check_dataset.py
from src.config import ExperimentConfig
from src.dataset import load_countdown_dataset, format_prompt
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

from src.config import DatasetConfig
cfg = DatasetConfig(name="Jiayi-Pan/Countdown-Tasks-3to4", max_eval_samples=5, max_train_samples=5)
train, _ = load_countdown_dataset(cfg)

import functools
fmt = functools.partial(format_prompt, tokenizer=tok)
train = train.map(fmt)

print(train[0]["prompt"])