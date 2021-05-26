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
from BartForDataToTextGeneration import BartForDataToText
import math
import random
import re
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from Data2TextProcessor_1 import SummaryDataModule
#from transformers.modeling_bart import shift_tokens_right

learning_rate = 3e-4 
max_epochs = 25

logger = TensorBoardLogger('tb_logs', name='my_model_epoch%s_%s'%(str(max_epochs), str(learning_rate)))


train_count = 0
val_count = 0

import os
import pytorch_lightning as pl

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

    
def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False


class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, freeze_encoder, freeze_embeds, eval_beams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model._make_duplicate_encoders()
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.freeze_embeds_ = freeze_embeds
        #self.hparams = hparams
        #self.hparams.update(hparams)

        if self.freeze_encoder:
            freeze_params(self.model.encoder)
            freeze_params(self.model.encoder1)
            freeze_params(self.model.encoder2)
            freeze_params(self.model.encoder3)
            freeze_params(self.model.encoder4)


        if freeze_embeds:
            self.freeze_embeds()
        self.save_hyperparameters()
  
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.shared)
        for d in [self.model.encoder, self.model.encoder1, self.model.encoder2,
            self.model.encoder3, self.model.encoder4, self.model.decoder]:
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
    
        input_ids_col0 = batch[0] if len(batch) >1 else None
        attention_mask_col0 = batch[1] if len(batch) >1 else None

        input_ids_col1 = batch[2] if len(batch) >3 else None
        attention_mask_col1 = batch[3] if len(batch) >3 else None

        input_ids_col2 = batch[4] if len(batch) >5 else None
        attention_mask_col2 = batch[5] if len(batch) >5 else None

        input_ids_col3 = batch[6] if len(batch) >7 else None
        attention_mask_col3 = batch[7] if len(batch) >7 else None

        input_ids_col4 = batch[8] if len(batch) >9 else None
        attention_mask_col4 = batch[9] if len(batch) >9 else None
        
        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        # Run the model and get the logits
        outputs = self(
            input_ids_col0 = input_ids_col0,
            input_ids_col1 = input_ids_col1,
            input_ids_col2 = input_ids_col2, 
            input_ids_col3 = input_ids_col3,
            input_ids_col4 = input_ids_col4,
            attention_mask_col0 = attention_mask_col0,
            attention_mask_col1 = attention_mask_col1,
            attention_mask_col2 = attention_mask_col2,
            attention_mask_col3 = attention_mask_col3,
            attention_mask_col4 = attention_mask_col4,
            labels = tgt_ids,
            decoder_input_ids = None,
            use_cache = False,
            encoder_combination_type = 'linearize'
        )
        
        lm_logits = outputs[1]
        # Create the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("Train Loss", loss, self.current_epoch)
        epoch_dictionary={
            'loss': loss,
            'log': tensorboard_logs}
        return epoch_dictionary

    def validation_step(self, batch, batch_idx):
    
        
        input_ids_col0 = batch[0] if len(batch) >1 else None
        attention_mask_col0 = batch[1] if len(batch) >1 else None

        input_ids_col1 = batch[2] if len(batch) >3 else None
        attention_mask_col1 = batch[3] if len(batch) >3 else None

        input_ids_col2 = batch[4] if len(batch) >5 else None
        attention_mask_col2 = batch[5] if len(batch) >5 else None

        input_ids_col3 = batch[6] if len(batch) >7 else None
        attention_mask_col3 = batch[7] if len(batch) >7 else None

        input_ids_col4 = batch[8] if len(batch) >9 else None
        attention_mask_col4 = batch[9] if len(batch) >9 else None
    
        

        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        #decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        # Run the model and get the logits
        outputs = self(
            input_ids_col0 = input_ids_col0,
            input_ids_col1 = input_ids_col1,
            input_ids_col2 = input_ids_col2, 
            input_ids_col3 = input_ids_col3,
            input_ids_col4 = input_ids_col4,
            attention_mask_col0 = attention_mask_col0,
            attention_mask_col1 = attention_mask_col1,
            attention_mask_col2 = attention_mask_col2,
            attention_mask_col3 = attention_mask_col3,
            attention_mask_col4 = attention_mask_col4,
            labels = tgt_ids,
            decoder_input_ids = None,
            use_cache = False,
            encoder_combination_type = 'linearize'
        )


        lm_logits = outputs[1]
        #print("LM LOGITS", lm_logits)
        #print("TGT IDS", tgt_ids)

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        tensorboard_logs = {'val_loss': val_loss}
        self.logger.experiment.add_scalar("Val Loss", val_loss, self.current_epoch)
        epoch_dictionary={
            'loss': val_loss,
            'log': tensorboard_logs}
        #print(epoch_dictionary)
        return epoch_dictionary

def make_data(tokenizer, data_type = 'robo', path = '/home/sanjana'):
    if data_type == 'robo':
        train_file = path + '/roboreviewer_summarization/data/robo_train_field_sep.csv'
        dev_file = path + '/roboreviewer_summarization/data/robo_dev_field_sep.csv'
        test_file = path + '/roboreviewer_summarization/data/robo_test_field_sep.csv'
    
    elif data_type =='webnlg':
        train_file = path + '/roboreviewer_summarization/data/web_nlg_train.csv'
        dev_file = path + '/roboreviewer_summarization/data/web_nlg_dev.csv'
        test_file = path + '/roboreviewer_summarization/data/web_nlg_test.csv'

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1)
    summary_data.prepare_data()
    return summary_data


def main():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartForDataToText.from_pretrained('facebook/bart-base')    
    summary_data = make_data(tokenizer, path = '/home/sanjana')

    #hparams = argparse.Namespace()
    freeze_encoder = True
    freeze_embeds = True
    eval_beams = 4

    model = LitModel(learning_rate = learning_rate, tokenizer = tokenizer, model = bart_model, freeze_encoder = freeze_encoder, freeze_embeds = freeze_embeds, eval_beams = eval_beams)
    checkpoint = ModelCheckpoint('checkpoint_files/')
    trainer = pl.Trainer(gpus=2, accelerator='dp', 
			max_epochs = max_epochs,
                        min_epochs = 1,
                        auto_lr_find = False,
                        progress_bar_refresh_rate = 100,
                        logger=logger,
                        callbacks=[CheckpointEveryNSteps(save_step_frequency = 5)])

    trainer.fit(model, summary_data)
    trainer.save_checkpoint("robo_model_epoch%s_adam_%s_linearize.ckpt"%(str(learning_rate), str(max_epochs)))


if __name__ == '__main__': 
    main()
   
