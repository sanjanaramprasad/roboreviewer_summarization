import pandas as pd
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper, BartModel
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
import torch

import math
import random
import re
import argparse
##import pandas as pd

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=1024, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}

    for sentence in source_sentences:
        
        #print(sentence)
        encoded_dict = tokenizer(
          sentence,
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
          add_prefix_space = True
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    for sentence in target_sentences:
        encoded_dict = tokenizer(
          sentence,
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
          add_prefix_space = True
        )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)
  

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch




class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_files, batch_size, num_examples = 20000):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_files = data_files
        self.batch_size = batch_size
        self.num_examples = num_examples

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.train = pd.read_csv(self.data_files[0])
        self.validate = pd.read_csv(self.data_files[1])
        self.test = pd.read_csv(self.data_files[2])


    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
        #sprint(self.train)
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])
        #print(self.test)

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self, data_type = 'robo'):
        #dataset = TensorDataset
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'],
                                    self.train['labels'])
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self, data_type = 'robo'):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'],
                                    self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self, data_type = 'robo'):
        #print(self.test['punchline_text_ids'])
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'],
                                    self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data


if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', 
                                                    additional_special_tokens=["<study>",  "</study>",
                                                                                "<punchline_text>", "</punchline_text>",
                                                                                "<punchline_effect>", "</punchline_effect>",
                                                                                "<population>", "</population>",
                                                                                "<interventions>", "</interventions>",
                                                                                "<outcomes>", "</outcomes>"])
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/robo_train_linearized.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/robo_dev_linearized.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/robo_test_linearized.csv'], batch_size = 1)

    summary_data.prepare_data()
    summary_data.setup("stage")
    train_data = summary_data.train_dataloader()
    train_batches = iter(train_data)
    batch= next(train_batches)
    print(batch[1])
    print(batch[2])
    print(tokenizer.decode(batch[0][0]))
