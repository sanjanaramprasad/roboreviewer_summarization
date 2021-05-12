import transformers
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
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
from Data2TextProcessor import SummaryDataModule
from Data2TextGenerator import Data2TextGenerator

logger = TensorBoardLogger('tb_logs', name='my_model13')

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens



class LitModel(pl.LightningModule):
    # Instantiate the model
    def __init__(self, learning_rate, tokenizer, model, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.learning_rate = learning_rate
        # self.freeze_encoder = freeze_encoder
        # self.freeze_embeds_ = freeze_embeds
        self.hparams = hparams

        if self.hparams.freeze_encoder:
            freeze_params(self.model.encoder_punchline_texts)
            freeze_params(self.model.encoder_punchline_effects)
            freeze_params(self.model.encoder_populations)
            freeze_params(self.model.encoder_interventions)
            freeze_params(self.model.encoder_outcomes)


        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        self.save_hyperparameters()
  
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.shared)
        for d in [self.model.encoder_punchline_texts, self.model.encoder_punchline_texts, self.model.encoder_populations,
            self.model.encoder_interventions, self.model.encoder_outcomes, self.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids_punchline_texts, **kwargs):
        return self.model(input_ids_punchline_texts, **kwargs)
  
    def configure_optimizers(self):
        print("PARAMS", self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        #print(batch)
    
        src_punchline_text_ids, src_punchline_text_mask = batch[0], batch[1]
        src_punchline_effect_ids, src_punchline_effect_mask = batch[2], batch[3]
        src_population_ids, src_population_mask = batch[4], batch[5]
        src_interventions_ids, src_interventions_mask = batch[6], batch[7]
        src_outcomes_ids, src_outcomes_mask = batch[8], batch[9]
        
        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        # Run the model and get the logits
        outputs = self(
        input_ids_punchline_texts = src_punchline_text_ids,
        input_ids_punchline_effects = src_punchline_effect_ids,
        input_ids_populations = src_population_ids, 
        input_ids_interventions = src_interventions_ids,
        input_ids_outcomes = src_outcomes_ids,
        attention_mask_punchline_texts = src_punchline_text_mask,
        attention_mask_punchline_effects = src_punchline_effect_mask,
        attention_mask_populations = src_population_mask,
        attention_mask_interventions = src_interventions_mask,
        attention_mask_outcomes = src_outcomes_mask,
        decoder_input_ids=decoder_input_ids,
            use_cache = False
        )
        lm_logits = outputs[0]
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
    
        '''src_punchline_text_ids, src_punchline_text_mask = batch[0], batch[1]
        src_outcome_ids, src_outcome_mask = batch[2], batch[3]
        src_punchline_effect_ids, src_punchline_effect_mask = batch[4], batch[5]
        src_population_ids, src_population_mask = batch[6], batch[7]
        src_interventions_ids, src_interventions_mask = batch[8], batch[9]
        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[10]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
    
        # Run the model and get the logits
        outputs = self(
        input_ids_punchline_texts = src_punchline_text_ids,
        input_ids_punchline_effects = src_punchline_effect_ids,
        input_ids_populations = src_population_ids, 
        input_ids_interventions = src_interventions_ids,
        input_ids_outcomes = src_outcome_ids,
        attention_mask_punchline_texts = src_punchline_text_mask,
        attention_mask_punchline_effects = src_punchline_effect_mask,
        attention_mask_populations = src_population_mask,
        attention_mask_interventions = src_interventions_mask,
        attention_mask_outcomes = src_outcome_mask,
        decoder_input_ids=decoder_input_ids,
            use_cache = False
        )'''
        src_punchline_text_ids, src_punchline_text_mask = batch[0], batch[1]
        src_punchline_effect_ids, src_punchline_effect_mask = batch[2], batch[3]
        src_population_ids, src_population_mask = batch[4], batch[5]
        src_interventions_ids, src_interventions_mask = batch[6], batch[7]
        src_outcomes_ids, src_outcomes_mask = batch[8], batch[9]

        # Load the data into variables
        #src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[-1]
        # Shift the decoder tokens right (but NOT the tgt_ids)
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        # Run the model and get the logits
        outputs = self(
        input_ids_punchline_texts = src_punchline_text_ids,
        input_ids_punchline_effects = src_punchline_effect_ids,
        input_ids_populations = src_population_ids,
        input_ids_interventions = src_interventions_ids,
        input_ids_outcomes = src_outcomes_ids,
        attention_mask_punchline_texts = src_punchline_text_mask,
        attention_mask_punchline_effects = src_punchline_effect_mask,
        attention_mask_populations = src_population_mask,
        attention_mask_interventions = src_interventions_mask,
        attention_mask_outcomes = src_outcomes_mask,
        decoder_input_ids=decoder_input_ids,
            use_cache = False
        )


        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        tensorboard_logs = {'val_loss': val_loss}
        self.logger.experiment.add_scalar("Val Loss", val_loss, self.current_epoch)
        epoch_dictionary={
            'loss': val_loss,
            'log': tensorboard_logs}
        #print(epoch_dictionary)
        return epoch_dictionary

  
    # Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, all_input_ids, input_ids_punchline_text, input_ids_outcomes, input_ids_punchline_effect,  \
        input_ids_population, input_ids_interventions, eval_beams, early_stopping = True, max_len = 40):
        ''' Function to generate text '''
        model_kwargs = self.model.generate_kwargs(
            all_input_ids = all_input_ids,
            input_ids_punchline_text = input_ids_punchline_text,
            input_ids_outcomes = input_ids_punchline_text,
            input_ids_punchline_effect = input_ids_punchline_effect,
            input_ids_population = input_ids_population,
            input_ids_interventions = input_ids_interventions,
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            early_stopping = early_stopping
        )
        input_ids = torch.ones((4, 1), device=self.model.device, dtype=torch.long)
        input_ids = input_ids * self.model.config.decoder_start_token_id
        beam_scorer = BeamSearchScorer(batch_size=1, max_length=self.model.config.max_length, num_beams=4, device=self.model.device, )
        logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=self.model.config.eos_token_id),])
        logits_warper = LogitsProcessorList([TopKLogitsWarper(50),TemperatureLogitsWarper(0.7),])
        outputs = self.model.beam_sample(input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs)
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in outputs]

def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False


def main():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartForDataToText.from_pretrained('facebook/bart-base')    
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/d2t_summarization/data/robo_train_field_sep.csv', 
                                           '/home/sanjana/d2t_summarization/data/robo_dev_field_sep.csv', 
                                           '/home/sanjana/d2t_summarization/data/robo_test_field_sep.csv'], batch_size = 1)

    summary_data.prepare_data()
    hparams = argparse.Namespace()

    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4
    model = LitModel(learning_rate = 3e-4, tokenizer = tokenizer, model = bart_model, hparams = hparams)
    checkpoint = ModelCheckpoint('checkpoint_files_2/')
    trainer = pl.Trainer(gpus=2, accelerator='dp',
                        max_epochs = 15,
                        min_epochs = 1,
                        auto_lr_find = False,
                        checkpoint_callback = checkpoint,
                        progress_bar_refresh_rate = 100,
                        logger=logger)

    trainer.fit(model, summary_data)
    trainer.save_checkpoint("example_sample_concat.ckpt")


if __name__ == '__main__': 
    main()
