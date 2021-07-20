import pandas as pd
import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
#import pandas as pd
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

    encoded_sentences = {}

    target_ids = []
    
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
    
    def get_encoding(snippet, key):
        #print(snippet)
        if isinstance(snippet, list):
            snippet_processed = []
            for ind, each in enumerate(snippet):
                enc = run_bart(each)
                if True:
                    key = "study"
                    #each = "<%s> "%key + each+" </%s>"%key
                    #study_key = "study%s"%(str(ind))
                    each = "<%s> "%key + each + " </%s> "%key
                    snippet_processed.append(each)
            snippet = " ".join(snippet_processed)
        #print(snippet)
        encoded_dict = run_bart(snippet)
        return encoded_dict
    
    def pad_sentences(ids, type = "sentence"):
        if ids.shape[0] < 20:
            num_sentences = ids.shape[0]
            filler = [1] * 512
            if type == "attention":
                filler = [0] * 512
            padded_sentences = [filler for i in range(0, 20 - num_sentences)]
            padded_sentences = torch.as_tensor(padded_sentences)
            padded_ids = torch.cat([ids ,padded_sentences ])
            #print(padded_ids)
            return padded_ids
        return ids
        
    sentence_dict_len = 0

    sentence_dict = eval(source_sentences[0])
    sentence_keys = list(sentence_dict.keys())
    sentence_keys_map = { key : 'col%s'%(str(i)) for i, key in enumerate(sentence_keys) }
    reverse_map = {v : k for k , v in sentence_keys_map.items()}
    #print(sentence_keys_map)
    for sentence, tgt_sentence in list(zip(source_sentences, target_sentences)):
        
        sentence_dict = eval(sentence)
        #sentence_dict = json.loads(sentence.replace("\'", "\""))
        #print(sentence_dict)
        sentence_dict = {sentence_keys_map[key] : val for key, val in sentence_dict.items()}
        #print(sentence_dict)
        sentence_dict_len = len(list(sentence_dict.keys()))
        keys = list(sentence_dict.keys())
        if True:
            for i in range(0, sentence_dict_len):
                keys_ids = 'ids_col%s'%(str(i))
                attention_masks_ids = 'attention_masks_col%s'%(str(i))

                if keys_ids not in encoded_sentences:
                    encoded_sentences[keys_ids] = []
                if attention_masks_ids not in encoded_sentences:
                    encoded_sentences[attention_masks_ids] = []

                #print(sentence_dict['col%s'%(str(i))])

                sentence_encoding = get_encoding(sentence_dict['col%s'%(str(i))], reverse_map['col%s'%(str(i))] )
                encoded_sentences[keys_ids].append(sentence_encoding['input_ids'])
                encoded_sentences[attention_masks_ids].append(sentence_encoding['attention_mask'])

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

    for i in range(0, sentence_dict_len):
        keys_ids = 'ids_col%s'%(str(i))
        attention_masks_ids = 'attention_masks_col%s'%(str(i))
        #print(encoded_sentences[keys_ids])
        encoded_sentences[keys_ids] = torch.cat(encoded_sentences[keys_ids], dim = 0)
        #print(encoded_sentences[keys_ids])
        encoded_sentences[attention_masks_ids] = torch.cat(encoded_sentences[attention_masks_ids], dim = 0)

    target_ids = torch.cat(target_ids, dim = 0)
    
    encoded_sentences['labels'] = target_ids

    return encoded_sentences




class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_files, batch_size, num_examples = 20000 , max_len = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_files = data_files
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.max_len = max_len

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.train = pd.read_csv(self.data_files[0])
        self.validate = pd.read_csv(self.data_files[1])
        self.test = pd.read_csv(self.data_files[2])

    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'], max_length = self.max_len)
        #sprint(self.train)
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'], max_length = self.max_len)
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'], max_length = self.max_len)
        #print(self.test)

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self, data_type = 'robo'):
        #dataset = TensorDataset
        if data_type == 'robo':
            dataset = TensorDataset(self.train['ids_col0'], self.train['attention_masks_col0'],
                                    self.train['ids_col1'], self.train['attention_masks_col1'],
                                    self.train['ids_col2'], self.train['attention_masks_col2'],
                                    self.train['ids_col3'], self.train['attention_masks_col3'],
                                    self.train['ids_col4'], self.train['attention_masks_col4'],
                                    self.train['labels'])
        elif data_type == 'webnlg':
            dataset = TensorDataset(self.train['ids_col0'], self.train['attention_masks_col0'],
                                    self.train['ids_col1'], self.train['attention_masks_col1'],
                                    self.train['ids_col2'], self.train['attention_masks_col2'],
                                    self.train['labels'])
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self, data_type = 'robo'):
        if data_type == 'robo':
            dataset = TensorDataset(self.validate['ids_col0'], self.validate['attention_masks_col0'],
                                    self.validate['ids_col1'], self.validate['attention_masks_col1'],
                                    self.validate['ids_col2'], self.validate['attention_masks_col2'],
                                    self.validate['ids_col3'], self.validate['attention_masks_col3'],
                                    self.validate['ids_col4'], self.validate['attention_masks_col4'],
                                    self.validate['labels'])
        elif data_type == 'webnlg':
            dataset = TensorDataset(self.validate['ids_col0'], self.validate['attention_masks_col0'],
                                    self.validate['ids_col1'], self.validate['attention_masks_col1'],
                                    self.validate['ids_col2'], self.validate['attention_masks_col2'],
                                    self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self, data_type = 'robo'):
        #print(self.test['punchline_text_ids'])
        if data_type == 'robo':
            dataset = TensorDataset(self.test['ids_col0'], self.test['attention_masks_col0'],
                                    self.test['ids_col1'], self.test['attention_masks_col1'],
                                    self.test['ids_col2'], self.test['attention_masks_col2'],
                                    self.test['ids_col3'], self.test['attention_masks_col3'],
                                    self.test['ids_col4'], self.test['attention_masks_col4'],
                                    self.test['labels'])
        elif data_type == 'webnlg':
            dataset = TensorDataset(self.test['ids_col0'], self.test['attention_masks_col0'],
                                    self.test['ids_col1'], self.test['attention_masks_col1'],
                                    self.test['ids_col2'], self.test['attention_masks_col2'],
                                    self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data


if __name__ == '__main__':
    additional_special_tokens = ["<attribute>", "</attribute>", "<sep>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")
    tokenizer.add_tokens(additional_special_tokens)
    #bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/bart_multienc_per_key/robo_train_sep.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/bart_multienc_per_key/robo_dev_sep.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/bart_multienc_per_key/robo_test_sep.csv'], batch_size = 1)
    summary_data.prepare_data()
    summary_data.setup("stage")
    it = summary_data.train_dataloader()
    batches = iter(it)
    batch = next(batches)


    generated_ids = batch[0]
    output = " ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids])
    print(output)
    '''file_contents = pd.read_csv('/home/sanjana/roboreviewer_summarization/data/bart_multienc_per_key/robo_dev_sep.csv')
    source_contents = file_contents[0]
    source_contents = eval(source_contents['source']) 
    index = 0
    print(batch)
    for key, values in source_contents.items():
        print(key)
        source_values = ["<study> "+ each + " </study>" for each in values]
        source_values = " ".join(source_values)
        print(source_values)
        

    
    
    #print(next(batch))'''
