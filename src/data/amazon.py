from typing import Literal, Optional
import os
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
import json


class AmazonDataset(Dataset):
    def __init__(
        self, 
        sentiment: Literal['positive', 'negative', 'all'],
        data_dir: str, 
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        max_length: Optional[int] = None,
        split: Literal['train', 'eval', 'test', 'all'] = 'train',
    ):
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        if tokenizer is not None:
            assert max_length is not None, 'max_length should be set if tokenizer is set!'

        self.file_path = os.path.join(data_dir, 'amazon', f'amazon_small_{split}.jsonl')
        
        self.file_positions = []
        with open(self.file_path, 'r') as f:
            pos, line = f.tell(), f.readline()
            while line:
                data = json.loads(line)
                if sentiment == 'positive' and data['sentiment'] == 'positive':
                    self.file_positions.append(pos)
                elif sentiment == 'negative' and data['sentiment'] == 'negative':
                    self.file_positions.append(pos)
                elif sentiment == 'all':
                    self.file_positions.append(pos)
                pos, line = f.tell(), f.readline()
    
    def __len__(self):
        return len(self.file_positions)

    def __getitem__(self, idx):
        file_pos = self.file_positions[idx]
        with open(self.file_path, 'r') as f:
            f.seek(file_pos)
            line = f.readline()
            data = json.loads(line)

        text = data['text']
        if self.tokenizer is not None:
            text = self.tokenizer.encode(
                text=text, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt',
                return_token_type_ids=False,
                return_attention_mask=False,
            ).squeeze() # type: ignore
            
        return text