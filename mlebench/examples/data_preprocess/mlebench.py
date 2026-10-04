"""
Create dataset (prompts) for each MLE-Bench task, saved in parquet format.
"""

import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/countdown',
                        help="path to store the preprocessed data locally")
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=32768)
    parser.add_argument('--test_size', type=int, default=128)
    parser.add_argument('--competition_id', type=str, default='spaceship-titanic')
    parser.add_argument('--prompt-path', required=True, type=str)
    parser.add_argument('--self-improve-prompt-path', type=str, default=None)

    args = parser.parse_args()

    data_source = args.competition_id
    print(f"Competition ID: {args.competition_id}")
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Placeholder rows: every row carries the same competition prompt; only the row index is used.
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    prompt = open(args.prompt_path).read()
    self_improve_prompt = open(args.self_improve_prompt_path).read()
    # prompts reference the prepared mle-bench data tree through a single
    # placeholder, resolved here at build time.
    _data_root = os.environ.get("MLE_BENCH_DATA", "/path/to/mle-bench")
    prompt = prompt.replace("${MLE_BENCH_DATA}", _data_root)
    self_improve_prompt = self_improve_prompt.replace("${MLE_BENCH_DATA}", _data_root)

    def make_map_fn(split):
        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                    "self-improve": self_improve_prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": args.competition_id
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Preprocessed data saved to {local_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
