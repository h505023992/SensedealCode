import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class Seq2SeqDataset(Dataset):
    """
    A Simple Seq2Seq Dataset Implementation
    """
    def __init__(self, fact_filename, romantic_filename,funny_filename, tokenizer, add_bos_token=True, add_eos_token=True):
        data = []
        with open(fact_filename, 'r') as f:
            line = f.readline()
            while line:
                data.append({"source": "", "target": line.replace('\n', ''), "style": "fact"})
                line = f.readline()

        with open(romantic_filename, 'r') as f:
            line = f.readline()
            while line:
                data.append({"source": "", "target": line.replace('\n', ''), "style": "romantic"})
                line = f.readline()

        with open(funny_filename, 'r') as f:
            line = f.readline()
            while line:
                data.append({"source": "", "target": line.replace('\n', ''), "style": "funny"})
                line = f.readline()

        self.data = data
        self.tokenizer = tokenizer
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def __getitem__(self, index):
        item = self.data[index]
        #source_token_ids = self.tokenizer.encode(item["source"], add_special_tokens=False)
        target_token_ids = self.tokenizer.encode(item["target"], add_special_tokens=False)
        if self.add_bos_token:
            target_token_ids.insert(0, self.tokenizer.bos_token_id)

        if self.add_eos_token:
            target_token_ids.append(self.tokenizer.eos_token_id)
        item["target_token_ids"] = torch.LongTensor(target_token_ids)

        if item["style"] == 'fact':
            item["source_token_ids"] = [1, 0, 0]
        elif item["style"] == 'romantic':
            item["source_token_ids"] = [0, 1, 0]
        elif item["style"] == 'funny':
            item["source_token_ids"] = [0, 0, 1]

        return item

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        new_batch = {}
        new_batch["source_token_ids"] = torch.tensor([item["source_token_ids"] for item in batch])
        new_batch["target_token_ids"] = pad_sequence(
            [item["target_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        new_batch["style"] = [item["style"] for item in batch]
        # sample_batch_size = len(new_batch["target_token_ids"])
        # past = torch.randn(size=(12, 2, sample_batch_size, 12, 1, 61))  # .to(self.device)  # 61=64-3
        # temp = new_batch["source_token_ids"].unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
        # classification = torch.tile(temp, (12, 2, 1, 12, 1, 1))
        # new_batch["past"] = torch.cat((classification, past), dim=-1)
        sample_batch_size = len(new_batch["target_token_ids"])
        extend_source = torch.zeros((sample_batch_size,1))
        temp =  torch.cat((new_batch["source_token_ids"], extend_source ), dim=1)
        temp = temp.unsqueeze(1).unsqueeze(2).unsqueeze(0).unsqueeze(0)
        new_batch["past"] = torch.tile(temp, (12, 2, 1, 12, 1, 16))
        return new_batch


