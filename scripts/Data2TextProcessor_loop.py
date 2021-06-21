import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper, BartModel
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
import torch
from torch import nn

import math
import random
import re
import argparse
global max_list_len 
max_list_len =0

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

#max_list_len = 0
def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=256, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}
    max_list_len = 0
    for sentence_list in source_sentences:
        sentence_list = eval(sentence_list)
        sentence_ids = []
        sentence_masks = []
        for sentence in sentence_list:
            #print(sentence)
            encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
            )
            #print(encoded_dict['input_ids'].shape)
            sentence_ids.append(encoded_dict['input_ids'])
            sentence_masks.append(encoded_dict['attention_mask'])
        #if len(sentence_ids) > max_list_len:
        #max_list_len =  len(sentence_ids)
        
        sentence_ids = torch.cat(sentence_ids, dim = 1)
        #print(sentence_ids.shape)
        if sentence_ids.shape[1] > max_list_len:
            max_list_len = sentence_ids.shape[1]
        sentence_masks = torch.cat(sentence_masks, dim = 1)
        input_ids.append(sentence_ids)
        attention_masks.append(sentence_masks)

    input_ids_padded = []
    attention_masks_padded = []
    for inp in input_ids:
        inp_padded = nn.ConstantPad1d((0, max_list_len - inp.shape[1]),-2)(inp)
        #print(inp_padded.shape)
        input_ids_padded.append(inp_padded)
    
    for attn in attention_masks:
        attn_padded = nn.ConstantPad1d((0, max_list_len - attn.shape[1]),-2)(attn)
        attention_masks_padded.append(attn_padded)

    print(max_list_len)
    #print(input_ids_padded[0].shape)
    input_ids = torch.cat(input_ids_padded, dim=0)
    attention_masks = torch.cat(attention_masks_padded, dim=0)

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
        self.train = pd.read_csv(self.data_files[0])[:20]
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
                                                    additional_special_tokens=["<study>", "<sep>", "</study>",
                                                                                "<punchline_text>", "</punchline_text>",
                                                                                "<punchline_effect>", "</punchline_effect>",
                                                                                "<population>", "</population>",
                                                                                "<interventions>", "</interventions>",
                                                                                "<outcomes>", "</outcomes>"])
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/robo_train_linearized_per_study.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/robo_dev_linearized_per_study.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/robo_test_linearized_per_study.csv'], batch_size = 1)

    summary_data.prepare_data()
    summary_data.setup("stage")
    train_data = summary_data.train_dataloader()
    train_batches = iter(train_data)
    batch= next(train_batches)
    current_ind = 0
    print(batch)
    source = batch[0]
    print(source.shape)
    #print(max_list_len) 
    print(source.shape[1]) 
    #print(tokenizer.decode(source[0][:1024]))
    for i in range(0, source.shape[1], 1024):
        chunk = source[:,i : i+ 1024]
        if chunk[0][0] != -2 and chunk[0][0] != 1:
            #print(chunk)
            print(tokenizer.decode(chunk[0]))
    print('=' * 13)
    #print(source[:,i : i+ 1024].shape)
    #print(batch[1])
    #print(batch[2])
