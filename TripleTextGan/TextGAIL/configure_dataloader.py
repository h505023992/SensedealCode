import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from seq2seq_dataset import Seq2SeqDataset

# pylint:disable=no-member


# class TextRLCollator:
#     def __init__(self, config, tokenizer):
#         self.config = config
#         self.tokenizer = tokenizer
#
#     def sample_collate(self, batch):
#         new_batch = {}
#         new_batch["source_text"] = [item["source_text"] for item in batch]
#         new_batch["target_text"] = [item["target_text"] for item in batch]
#         new_batch["source_token_ids"] = pad_sequence(
#             [item["source_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         new_batch["target_token_ids"] = pad_sequence(
#             [item["target_token_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         return new_batch


def lazy_collate(batch):
    return batch


class DataLoaderHandler:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        # self.collator = Collator(config, self.tokenizer)

    def train_dataloader(self,config):
        fact_train_filename = os.path.join(self.config.task.data_dir, f"fact-train.txt")
        romantic_train_filename = os.path.join(self.config.task.data_dir, f"romantic-train.txt")
        funny_train_filename = os.path.join(self.config.task.data_dir, f"funny-train.txt")
        train_dataset = Seq2SeqDataset(fact_train_filename,romantic_train_filename,funny_train_filename , self.tokenizer)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.training.batch_size, shuffle=True, collate_fn=lazy_collate)
        return train_dataloader

    def valid_dataloader(self,config):
        fact_valid_filename = os.path.join(self.config.task.data_dir, f"fact-val.txt")
        romantic_valid_filename = os.path.join(self.config.task.data_dir, f"romantic-val.txt")
        funny_valid_filename = os.path.join(self.config.task.data_dir, f"funny-val.txt")
        val_dataset = Seq2SeqDataset(fact_valid_filename,romantic_valid_filename,funny_valid_filename, self.tokenizer)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, collate_fn=lazy_collate)
        return val_dataloader

    def test_dataloader(self,config):
        fact_test_filename = os.path.join(self.config.task.data_dir, f"fact-test.txt")
        romantic_test_filename = os.path.join(self.config.task.data_dir, f"romantic-test.txt")
        funny_test_filename = os.path.join(self.config.task.data_dir, f"funny-test.txt")
        test_dataset = Seq2SeqDataset(fact_test_filename,romantic_test_filename,funny_test_filename, self.tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, collate_fn=lazy_collate)
        return test_dataloader