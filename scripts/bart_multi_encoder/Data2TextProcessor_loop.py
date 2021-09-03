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


def encode_sentences(tokenizer, df, source_keys, targets, max_length=1024, pad_to_max_length=True, return_tensors="pt"):

    encoded_sentences = {}
    max_list_len = 0
    target_ids = []
    input_ids = []
    attention_masks = []
    def run_bart(snippet):
        encoded_dict = tokenizer(
          snippet,
          max_length=max_length,
          padding="max_length" if pad_to_max_length else None,
          truncation=True,
          return_tensors=return_tensors,
          add_prefix_space = True
        )
        return encoded_dict

    def get_encoding(snippets, key):
        #print(snippet)
        snippet_processed = []
        for each in snippets:
                each = each.strip()
                each = "<study> <%s> "%key + each+" </%s> </study>"%key
                snippet_processed.append(each)
        snippet = " ".join(snippet_processed)
        #print(snippet)
        encoded_dict = run_bart(snippet.strip())
        return encoded_dict


    for _, row in df.iterrows():
        #print(row)
        studies = {}
        num_studies = 0
        for key in source_keys:
            #print(row[key])
            key_sents = row[key]
            key_sents = eval(key_sents)
            key_sents = key_sents[:13]
            num_studies = len(key_sents)
            studies[key] = key_sents
        
        all_keys = list(studies.keys())
        #print(num_studies, studies)
        encoded_dict_temp = {}
        for i in range(0, num_studies):
            for k in all_keys:
                if "%s_ids"%k not in encoded_dict_temp:
                    encoded_dict_temp["%s_ids"%k] = []
                    encoded_dict_temp["%s_mask"%k] = []
                k_val = studies[k][i]
                encoded_dict = tokenizer(
                    k_val,
                    max_length=max_length,
                    padding="max_length" if pad_to_max_length else None,
                    truncation=True,
                    return_tensors=return_tensors,
                    add_prefix_space = True
                    )
                #print(k_val)
                encoded_dict_temp["%s_ids"%k].append(encoded_dict['input_ids'])
                encoded_dict_temp["%s_mask"%k].append(encoded_dict['attention_mask'])
                #print(encoded_dict_temp)

        for k, v in encoded_dict_temp.items():
                sentence_ids = torch.cat(encoded_dict_temp[k], dim = 1)
                #print(sentence_ids.shape)
                if sentence_ids.shape[1] > max_list_len:
                    max_list_len = sentence_ids.shape[1]
                if k not in encoded_sentences:
                    encoded_sentences[k] = []
                encoded_sentences[k].append(sentence_ids)
            
    
    for k in encoded_sentences:
        input_ids_padded = []
        attention_masks_padded = []
        inps = encoded_sentences[k]
        for inp in inps:
            inp_padded = nn.ConstantPad1d((0, max_list_len - inp.shape[1]),-2)(inp)
            input_ids_padded.append(inp_padded)
        input_ids_padded = torch.cat(input_ids_padded, dim=0)
        print(k, input_ids_padded.shape)
        encoded_sentences[k] = input_ids_padded
    
    for tgt_sentence in targets:
        encoded_dict = tokenizer(
              tgt_sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
        )
        # Shift the target ids to the right
        #shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])
    
        
    target_ids = torch.cat(target_ids, dim = 0)
    
    encoded_sentences['labels'] = target_ids
    #print(encoded_sentences['labels'].shape)

    return encoded_sentences

'''def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=256, pad_to_max_length=True, return_tensors="pt"):
     Function that tokenizes a sentence 
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}
    max_list_len = 0
    for sentence_list in source_sentences:
        sentence_list = eval(sentence_list)
        sentence_ids = []
        sentence_masks = []
        sentence_list = sentence_list[:10]
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

    return batch'''


def preprocess_df(df, keys):
    for key in keys:
        df = df[df[key] != "['']"]
    return df



class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_files, batch_size, num_examples = 20000 , max_len = 1024, flatten_studies = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_files = data_files
        print(self.data_files)
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.max_len = max_len
        self.flatten_studies = flatten_studies

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.train = pd.read_csv(self.data_files[0])
        self.validate = pd.read_csv(self.data_files[1])
        self.test = pd.read_csv(self.data_files[2])
        preprocess_keys = ['population', 'interventions', 'outcomes', 'SummaryConclusions','punchline_text', 'punchline_effect' ]
        self.train = preprocess_df(self.train, preprocess_keys)
        self.validate = preprocess_df(self.validate, preprocess_keys)
        self.test = preprocess_df(self.test, preprocess_keys)


    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, 
                                      self.train,
                                        ['population_mesh', 
                                        'interventions_mesh',
                                        'outcomes_mesh',
                                        'punchline_text',
                                        'punchline_effect'], 
                                        self.train['SummaryConclusions'],
                                        max_length = self.max_len)
        self.validate = encode_sentences(self.tokenizer, 
                                        self.validate,
                                        ['population_mesh', 
                                        'interventions_mesh',
                                        'outcomes_mesh',
                                        'punchline_text',
                                        'punchline_effect'], 
                                        self.validate['SummaryConclusions'],
                                        max_length = self.max_len)
        self.test = encode_sentences(self.tokenizer, 
                                    self.test,
                                        ['population_mesh', 
                                        'interventions_mesh',
                                        'outcomes_mesh',
                                        'punchline_text',
                                        'punchline_effect'], 
                                        self.test['SummaryConclusions'],
                                        max_length = self.max_len)

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self, data_type = 'robo'):
        dataset = TensorDataset(self.train['population_mesh_ids'], self.train['population_mesh_mask'],
                                self.train['interventions_mesh_ids'], self.train['interventions_mesh_mask'],
                                self.train['outcomes_mesh_ids'], self.train['outcomes_mesh_mask'],
                                self.train['punchline_text_ids'], self.train['punchline_text_mask'],
                                self.train['punchline_effect_ids'], self.train['punchline_effect_mask'],
                                    self.train['labels'])
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self, data_type = 'robo'):
        dataset = TensorDataset(self.validate['population_mesh_ids'], self.validate['population_mesh_mask'],
                                self.validate['interventions_mesh_ids'], self.validate['interventions_mesh_mask'],
                                self.validate['outcomes_mesh_ids'], self.validate['outcomes_mesh_mask'],
                                self.validate['punchline_text_ids'], self.validate['punchline_text_mask'],
                                self.validate['punchline_effect_ids'], self.validate['punchline_effect_mask'],
                                    self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self, data_type = 'robo'):
        #print(self.test['punchline_text_ids'])
        dataset = TensorDataset(self.test['population_mesh_ids'], self.test['population_mesh_mask'],
                                self.test['interventions_mesh_ids'], self.test['interventions_mesh_mask'],
                                self.test['outcomes_mesh_ids'], self.test['outcomes_mesh_mask'],
                                self.test['punchline_text_ids'], self.test['punchline_text_mask'],
                                self.test['punchline_effect_ids'], self.test['punchline_effect_mask'],
                                    self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data


def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/sanjana', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv'], max_len = 256):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    print(train_file)
    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = max_len, flatten_studies = True)
    summary_data.prepare_data()
    
    assert(len(summary_data.train) > 10)
    return summary_data

if __name__ == '__main__':
    additional_special_tokens = ["<sep>", "<study>", "</study>",
            "<outcomes_mesh>", "</outcomes_mesh>",
            "<punchline_text>", "</punchline_text>",
            "<population_mesh>", "</population_mesh>",
            "<interventions_mesh>", "</interventions_mesh>",
            "<punchline_effect>", "</punchline_effect>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens)
    #bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    data_files = ['train_rr_data.csv', 'dev_rr_data.csv' , 'test_rr_data.csv']

    
                                    
    
    summary_data = make_data(tokenizer, SummaryDataModule, data_type = 'robo', path = '/home/sanjana', files = data_files, max_len = 1024)
    

    summary_data.prepare_data()
    summary_data.setup()
    train_data = summary_data.train_dataloader()
    train_batches = iter(train_data)
    batch= next(train_batches)
    current_ind = 0
    print(batch)
    source = batch[0][0]
    print(source.shape)
    #print(max_list_len)  
    #print(tokenizer.decode(source[0][:1024]))
    for i in range(0, source.shape[0], 1024):
        chunk = source[i : i + 1024]
        print(i, chunk)
        if chunk[0] != -2 and chunk[0] != 1:
            #print(chunk)
            print(tokenizer.decode(chunk))
    print('=' * 13)
    #print(source[:,i : i+ 1024].shape)
    #print(batch[1])
    #print(batch[2])
