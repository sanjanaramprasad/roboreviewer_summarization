#import pandas as pd
from DataToTextProcessor import SummaryDataModule
import transformers
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
from model_latent import BartForDataToTextGeneration_MultiLM
import math
import random
import re
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
#from Data2TextProcessor_1 import SummaryDataModule
#from transformers.modeling_bart import shift_tokens_right

learning_rate = 3e-5 
max_epochs = 10

logger = TensorBoardLogger('tb_logs_final', name='my_model_final_epoch%s_%s_surface_content'%(str(max_epochs), str(learning_rate)))


train_count = 0
val_count = 0

import os
import pytorch_lightning as pl

def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False



def get_data(data):
        population_input_ids = data[0] 
        population_attention_masks = data[1] 

        interventions_input_ids = data[2] 
        interventions_attention_masks = data[3] 


        outcomes_input_ids = data[4] 
        outcomes_attention_masks = data[5] 

        punchline_text_input_ids = data[6] 
        punchline_text_attention_masks = data[7] 

        return population_input_ids, population_attention_masks,\
                interventions_input_ids, interventions_attention_masks,\
                outcomes_input_ids, outcomes_attention_masks,\
                punchline_text_input_ids, punchline_text_attention_masks,

class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, freeze_encoder, freeze_embeds, eval_beams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model.resize_token_embeddings(len(self.tokenizer))
        #self.model.model1.resize_token_embeddings(len(tokenizer))
        #self.model.model2.resize_token_embeddings(len(tokenizer))
        self.model._make_multiple_lm_heads()
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.freeze_embeds_ = freeze_embeds
        #self.hparams = hparams
        #self.hparams.update(hparams)

        if self.freeze_encoder:
            freeze_params(self.model.model.encoder)

        if freeze_embeds:
            self.freeze_embeds()
        #self.save_hyperparameters()
  
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids_col0, **kwargs):
        return self.model(input_ids_col0, **kwargs)
  
    def configure_optimizers(self):
        print("PARAMS", self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr= self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        #print(batch)
        population_input_ids, population_attention_masks,\
                interventions_input_ids, interventions_attention_masks,\
                outcomes_input_ids, outcomes_attention_masks,\
                punchline_text_input_ids, punchline_text_attention_masks = get_data(batch)

        
        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        # Run the model and get the logits
        outputs = self.model(
            input_ids_col0 = population_input_ids,
            input_ids_col1 = interventions_input_ids,
            input_ids_col2 = outcomes_input_ids, 
            input_ids_col3 = punchline_text_input_ids,
            attention_mask_col0 = population_attention_masks,
            attention_mask_col1 = interventions_attention_masks,
            attention_mask_col2 = outcomes_attention_masks,
            attention_mask_col3 = punchline_text_attention_masks,
            labels = tgt_ids,
            use_cache = False,
        )
        
        loss = outputs[0]
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("Train Loss", loss, self.current_epoch)
        epoch_dictionary={
            'loss': loss,
            'log': tensorboard_logs,
            }
        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
    
        
        population_input_ids, population_attention_masks,\
                interventions_input_ids, interventions_attention_masks,\
                outcomes_input_ids, outcomes_attention_masks,\
                punchline_text_input_ids, punchline_text_attention_masks = get_data(batch)
        

        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        #decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        # Run the model and get the logits
        outputs = self.model(
            input_ids_col0 = population_input_ids,
            input_ids_col1 = interventions_input_ids,
            input_ids_col2 = outcomes_input_ids, 
            input_ids_col3 = punchline_text_input_ids,
            attention_mask_col0 = population_attention_masks,
            attention_mask_col1 = interventions_attention_masks,
            attention_mask_col2 = outcomes_attention_masks,
            attention_mask_col3 = punchline_text_attention_masks,
            labels = tgt_ids,
            use_cache = False,
        )


        val_loss = outputs[0]

        tensorboard_logs = {'val_loss': val_loss}
        self.logger.experiment.add_scalar("Val Loss", val_loss, self.current_epoch)
        epoch_dictionary={
            'val_loss': val_loss,
            'log': tensorboard_logs}
        #print(epoch_dictionary)
        return epoch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}




def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/ramprasad.sa', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv']):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = 1024)
    summary_data.prepare_data()
    return summary_data


def main():

    additional_special_tokens = ['<population>', '</population>',
                                            '<interventions>', '</interventions>',
                                            '<outcomes>', '</outcomes>',
                                            '<punchline_text>', '</punchline_text>',
                                            '<punchline_effect>', '</punchline_effect>', "<sep>", "<bos>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",
                                                        eos_token="</s>",
                                                        pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens)
    #bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    data_files = ['train_rr_data.csv', 'dev_rr_data.csv' , 'test_rr_data.csv']

    
                                    
    
    
    summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/ramprasad.sa', files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv'])

    bart_model = BartForDataToTextGeneration_MultiLM.from_pretrained('facebook/bart-base') 

    #hparams = argparse.Namespace()
    freeze_encoder = False
    freeze_embeds = False
    eval_beams = 4

    model = LitModel(learning_rate = learning_rate, tokenizer = tokenizer, model = bart_model, freeze_encoder = freeze_encoder, freeze_embeds = freeze_embeds, eval_beams = eval_beams)
    checkpoint = ModelCheckpoint(dirpath = 'checkpoint_files_final/outputs_latent',
                                filename = '{epoch}-{val_loss:.2f}',
                                save_top_k=10,
                                monitor = 'val_loss')
    trainer = pl.Trainer(gpus=1,  
			accelerator='dp',
                        max_epochs = 3,
                        min_epochs = 1,
                        auto_lr_find = False,
                        progress_bar_refresh_rate = 100,
                        logger=logger,
                        callbacks=[checkpoint])

    trainer.fit(model, summary_data)
    ##trainer.save_checkpoint("robo_model_epoch%s_adam_%s_decomod.ckpt"%(str(learning_rate), str(max_epochs)))


if __name__ == '__main__': 
    main()
   

