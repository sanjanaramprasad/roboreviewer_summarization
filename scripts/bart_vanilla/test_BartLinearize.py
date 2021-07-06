
import transformers
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_utils import GenerationMixin
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import math
import random
import re
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from Data2TextProcessor_linearize import SummaryDataModule




class BartForConditionalGenerationTester():

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, batch):
        input_ids = batch[0]
        attention_mask = batch[1]
        tgt_ids = batch[2]

        outputs = self.model(input_ids = input_ids,
                attention_mask = attention_mask,
                labels = tgt_ids,
                decoder_input_ids = None,
                use_cache = False)
        optimizer = optim.Adam(self.model.parameters())
        lm_logits = outputs[1]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        #lm_logits = outputs[1]
        # Create the loss function
        #ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        #loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return outputs, loss 


def make_data(tokenizer, data_type = 'robo', path = '/home/sanjana'):
    if data_type == 'robo':
        train_file = path + '/roboreviewer_summarization/data/bart_vanilla/robo_train_linearized.csv'
        dev_file = path + '/roboreviewer_summarization/data/bart_vanilla/robo_train_linearized.csv'
        test_file = path + '/roboreviewer_summarization/data/bart_vanilla/robo_train_linearized.csv'

    elif data_type =='webnlg':
        train_file = path + '/roboreviewer_summarization/data/web_nlg_train.csv'
        dev_file = path + '/roboreviewer_summarization/data/web_nlg_dev.csv'
        test_file = path + '/roboreviewer_summarization/data/web_nlg_test.csv'

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1)
    summary_data.prepare_data()
    return summary_data





print("Init tokenizer ...")
additional_special_tokens = ["<study>", "</study>", "<outcomes>", "</outcomes>", "<punchline_text>", "</punchline_text>", "<population>", "</population>", "<interventions>", "</interventions>", "<punchline_effect>", "</punchline_effect>"]
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''additional_special_tokens=["<study>",  "</study>",
                                                                                "<punchline_text>", "</punchline_text>",
                                                                                "<punchline_effect>", "</punchline_effect>",
                                                                                "<population>", "</population>",
                                                                                "<interventions>", "</interventions>",
                                                                                "<outcomes>", "</outcomes>"])
'''
                                                    
print("Init model...")                                                    
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

tokenizer.add_tokens(additional_special_tokens)
bart_model.resize_token_embeddings(len(tokenizer))
obj = BartForConditionalGenerationTester(bart_model, tokenizer)
#bart_model.resize_token_embeddings(len(self.tokenizer))
print("Making data")
summary_data = make_data(tokenizer, path = '/home/sanjana')
summary_data.prepare_data()
summary_data.setup("stage")
train_data = summary_data.train_dataloader(data_type = 'robo')
it = iter(train_data)
batch = next(it)
batch = next(it)
batch = next(it)
print(batch[0].shape, batch[1].shape)
print(batch[1])
print(batch[2])
print(tokenizer.decode(batch[0][0]))
print(obj.forward(batch))

