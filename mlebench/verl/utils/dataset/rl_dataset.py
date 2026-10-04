# Copyright 2024 <org> Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.code_truncation import truncate_code_to_budget


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self._non_thinking_suffix = self._compute_non_thinking_suffix(tokenizer)

        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self._download()
        self._read_files_and_tokenize()


    @staticmethod
    def _compute_non_thinking_suffix(tokenizer) -> str:
        """
        Empty-think suffix that disables thinking mode for chat templates supporting
        `enable_thinking` (e.g. Qwen3); empty string otherwise. Needed because prompts
        are built as raw strings rather than via apply_chat_template.
        """
        template = getattr(tokenizer, 'chat_template', None) or ''
        return '\n<think>\n\n</think>\n\n' if 'enable_thinking' in template else ''

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)


    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = chat[0]['content'] + self._non_thinking_suffix
        # prompt_with_chat_template = chat

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


class SelfImproveDataset(RLHFDataset):
    def __init__(
        self, 
        parquet_files, 
        tokenizer, 
        batch_size,
        prompt_key='prompt', 
        max_prompt_length=1024, 
        filter_prompts=True, 
        cache_dir='~/.cache/verl/rlhf', 
        chat_template_func=None, 
        return_raw_chat=False, 
        truncation='error',
    ):
        super().__init__(parquet_files, tokenizer, prompt_key, max_prompt_length, filter_prompts, cache_dir, chat_template_func, return_raw_chat, truncation)
        self.prev_code_lst = None
        self.prev_error_lst = None
        self.batch_size = batch_size
        
        
    def set_prev_code_lst(self, lst):
        self.prev_code_lst = lst
        
    
    def set_prev_error_lst(self, error_lst):
        self.prev_error_lst = error_lst
        
        
    # Token slack for re-tokenization shifts when the code is substituted into the prompt.
    _TRUNCATION_SAFETY_MARGIN = 16

    def add_prev_code_to_prompt(self, prompt, idx):
        prev_code = self.prev_code_lst[idx]
        prev_error = self.prev_error_lst[idx]

        overhead = len(self.tokenizer(
            prompt.format(previous_plan_code='', previous_plan_error=prev_error)
            + self._non_thinking_suffix)['input_ids'])
        budget = self.max_prompt_length - overhead - self._TRUNCATION_SAFETY_MARGIN
        if budget > 0:
            prev_code = truncate_code_to_budget(prev_code, self.tokenizer, budget)
        else:
            prev_code = ''

        return prompt.format(previous_plan_code=prev_code, previous_plan_error=prev_error) + self._non_thinking_suffix
    
    
    def __len__(self):
        # One row per batch slot; prev_code_lst/prev_error_lst are indexed by slot.
        return self.batch_size
    
    
    def __getitem__(self, item):
        """
        Build the self-improvement prompt for row `item` from its previous code and error.
        """
        assert self.prev_code_lst is not None, "must have prev_code_lst!"
        assert self.prev_error_lst is not None, "must have prev_error_lst!"
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = chat[0]['self-improve']
        prompt_with_chat_template = self.add_prev_code_to_prompt(
            prompt=prompt_with_chat_template,
            idx=item,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict
    
    
    
