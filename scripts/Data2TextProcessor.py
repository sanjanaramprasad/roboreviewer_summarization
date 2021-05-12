import transformers
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper, BartModel
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


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

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=512, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
        Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    outcomes_ids = []
    attention_masks_outcomes = []

    punchline_text_ids = []
    attention_masks_punchline_text = []

    population_ids = []
    attention_masks_population = []

    interventions_ids = []
    attention_masks_interventions = []

    punchline_effect_ids = []
    attention_masks_punchline_effect = []

    target_ids = []
    tokenized_sentences = {}
    
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
    
    def get_encoding(snippet):
        #print(snippet)
        if isinstance(snippet, list):
            snippet_processed = []
            for each in snippet:
                enc = run_bart(each)
                if len(enc['input_ids']) < 40:
                    each = "<s> " + each+" </s>"
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
        
    
    for sentence, tgt_sentence in list(zip(source_sentences, target_sentences)):
        sentence_dict = eval(sentence)
        #sentence_dict = json.loads(sentence.replace("\'", "\""))
        if len(sentence_dict['outcomes']) <= 20:
            outcomes_dict = get_encoding(sentence_dict['outcomes'])
            #print(outcomes_dict['input_ids'].shape)
            #sentence_outcome_ids = outcomes_dict['input_ids']
            outcomes_ids.append(outcomes_dict['input_ids'])
            attention_masks_outcomes.append(outcomes_dict['attention_mask'])

            punchline_text_dict = get_encoding(sentence_dict['punchline_text'])              
            punchline_text_ids.append(punchline_text_dict['input_ids'])
            attention_masks_punchline_text.append(punchline_text_dict['attention_mask'])

            population_dict = get_encoding(sentence_dict['population'])              
            population_ids.append(population_dict['input_ids'])
            attention_masks_population.append(population_dict['attention_mask'])

            interventions_dict = get_encoding(sentence_dict['interventions'])               
            interventions_ids.append(interventions_dict['input_ids'])
            attention_masks_interventions.append(interventions_dict['attention_mask'])

            punchline_effect_dict = get_encoding(sentence_dict['punchline_effect'])               
            punchline_effect_ids.append(punchline_effect_dict['input_ids'])
            attention_masks_punchline_effect.append(punchline_effect_dict['attention_mask'])
            
            
            encoded_dict = tokenizer(
              tgt_sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
            )
            # Shift the target ids to the right
            shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
            target_ids.append(encoded_dict['input_ids'])
    
    outcomes_ids = torch.cat(outcomes_ids, dim = 0)
    punchline_text_ids = torch.cat(punchline_text_ids, dim = 0)
    population_ids = torch.cat(population_ids, dim = 0)
    interventions_ids = torch.cat(interventions_ids, dim = 0)
    punchline_effect_ids = torch.cat(punchline_effect_ids, dim = 0)

    attention_masks_outcomes = torch.cat(attention_masks_outcomes, dim = 0)
    attention_masks_punchline_text = torch.cat(attention_masks_punchline_text, dim = 0)
    attention_masks_population = torch.cat(attention_masks_population, dim = 0)
    attention_masks_interventions = torch.cat(attention_masks_interventions, dim = 0)
    attention_masks_punchline_effect = torch.cat(attention_masks_punchline_effect, dim = 0)
    target_ids = torch.cat(target_ids, dim = 0)
    

    batch = {
        "punchline_text_ids" : punchline_text_ids,
        "attention_masks_punchline_text" : attention_masks_punchline_text,
        "punchline_effect_ids" : punchline_effect_ids,
        "attention_masks_punchline_effect" : attention_masks_punchline_effect,
        "population_ids" : population_ids,
        "attention_masks_population" : attention_masks_population,
        "interventions_ids" : interventions_ids,
        "attention_masks_interventions" : attention_masks_interventions,
        "outcomes_ids" : outcomes_ids,
        "attention_masks_outcomes" : attention_masks_outcomes,
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
        #print(self.test)

    # encode the sentences using the tokenizer  
    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
        #sprint(self.train)
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])
        #print(self.test)
    
    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        #dataset = TensorDataset
        dataset = TensorDataset(self.train['punchline_text_ids'], self.train['attention_masks_punchline_text'],
                                self.train['punchline_effect_ids'], self.train['attention_masks_punchline_effect'],
                                self.train['population_ids'], self.train['attention_masks_population'],
                                self.train['interventions_ids'], self.train['attention_masks_interventions'],
                                self.train['outcomes_ids'], self.train['attention_masks_outcomes'],
                                self.train['labels'])
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.validate['punchline_text_ids'], self.validate['attention_masks_punchline_text'],
                                self.validate['punchline_effect_ids'], self.validate['attention_masks_punchline_effect'],
                                self.validate['population_ids'], self.validate['attention_masks_population'],
                                self.validate['interventions_ids'], self.validate['attention_masks_interventions'],
                                self.validate['outcomes_ids'], self.validate['attention_masks_outcomes'],
                                self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self):
        #print(self.test['punchline_text_ids'])
        dataset = TensorDataset(self.test['punchline_text_ids'], self.test['attention_masks_punchline_text'],
                                self.test['punchline_effect_ids'], self.test['attention_masks_punchline_effect'],
                                self.test['population_ids'], self.test['attention_masks_population'],
                                self.test['interventions_ids'], self.test['attention_masks_interventions'],
                                self.test['outcomes_ids'], self.test['attention_masks_outcomes'],
                                self.test['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data
