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
logger = TensorBoardLogger('tb_logs', name='my_model13')

'''def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens'''



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
            freeze_params(self.model.encoder_col0)
            freeze_params(self.model.encoder_col1)
            freeze_params(self.model.encoder_col2)
            freeze_params(self.model.encoder_col3)
            freeze_params(self.model.encoder_col4)


        if self.hparams.freeze_embeds:
            self.freeze_embeds()
        self.save_hyperparameters()
  
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.shared)
        for d in [self.model.encoder_col0, self.model.encoder_col1, self.model.encoder_col2,
            self.model.encoder_col3, self.model.encoder_col4, self.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids_col0, **kwargs):
        return self.model(input_ids_col0, **kwargs)
  
    def configure_optimizers(self):
        print("PARAMS", self.parameters())
        #optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr= self.learning_rate)
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
            use_cache = False
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
            use_cache = False
        )


        lm_logits = outputs[1]
        print("LM LOGITS", lm_logits)
        print("TGT IDS", tgt_ids)

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        tensorboard_logs = {'val_loss': val_loss}
        self.logger.experiment.add_scalar("Val Loss", val_loss, self.current_epoch)
        epoch_dictionary={
            'loss': val_loss,
            'log': tensorboard_logs}
        #print(epoch_dictionary)
        return epoch_dictionary


    def generate_text(self, 
        batch,
        num_beams = 4, 
        max_len = 13,
        ):

        generator = GenerationMixin()
        

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

        encoder_col0, encoder_col1, \
            encoder_col2, encoder_col3, encoder_col4 = self.model.get_encoders()


        logits_processor = generator._get_logits_processor(
            repetition_penalty=0.5,
            no_repeat_ngram_size=1,
            encoder_no_repeat_ngram_size=1,
            encoder_input_ids=input_ids_col0,
            min_length=5,
            max_length=13,
            num_beams=4,
            repetition_penalty=0.5,
            no_repeat_ngram_size=1,
            encoder_no_repeat_ngram_size=2,
            encoder_input_ids=input_ids_col0,
            bad_words_ids=None,
            min_length=5,
            max_length=12,
            eos_token_id=self.model.config.eos_token_id,
            forced_bos_token_id=self.model.config.forced_bos_token_id,
            forced_eos_token_id=self.model.config.forced_eos_token_id,
            prefix_allowed_tokens_fn=None,
            num_beams=4,
            num_beam_groups=4,
            diversity_penalty=0.5,
            remove_invalid_values=None,
        )

        model_kwargs = {}
        if not(input_ids_col0 is None):
            model_kwargs["encoder_outputs_col0"] = encoder_col0(input_ids_col0.repeat_interleave(num_beams, dim=0), return_dict=True)
        
        if not(input_ids_col1 is None):
            model_kwargs["encoder_outputs_col1"] = encoder_col1(input_ids_col1.repeat_interleave(num_beams, dim=0), return_dict=True)

        if not(input_ids_col2 is None):
            model_kwargs["encoder_outputs_col2"] = encoder_col2(input_ids_col2.repeat_interleave(num_beams, dim=0), return_dict=True)

        if not(input_ids_col3 is None):
            model_kwargs["encoder_outputs_col3"] = encoder_col3(input_ids_col3.repeat_interleave(num_beams, dim=0), return_dict=True)

        if not(input_ids_col4 is None):
            model_kwargs["encoder_outputs_col4"] = encoder_col4(input_ids_col4.repeat_interleave(num_beams, dim=0), return_dict=True)

        input_ids = torch.ones((num_beams, 1), device=self.model.device, dtype=torch.long)
        input_ids = input_ids * self.model.config.decoder_start_token_id

        stopping_criteria = StoppingCriteriaList()
        if max_len is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_len))
        
        beam_scorer = BeamSearchScorer(
                batch_size=1,
                num_beams=num_beams,
                device=self.model.device,
                length_penalty=2.0,
                do_early_stopping=True,
                stopping_criteria=stopping_criteria,
                **model_kwargs
            )
        #logits_processor = LogitsProcessorList([MinLengthLogitsProcessor(5, eos_token_id=self.model.config.eos_token_id),])
        outputs = self.model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)
        print("Generated:", self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
  
        return 

    '''# Method that generates text using the BartForConditionalGeneration's generate() method
    def generate_text(self, all_input_ids, input_ids_punchline_text, input_ids_outcomes, input_ids_punchline_effect,  \
        input_ids_population, input_ids_interventions, eval_beams, early_stopping = True, max_len = 40):
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
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in outputs]'''

def freeze_params(model):
    ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
        adapted from finetune.py '''
    for layer in model.parameters():
        layer.requires_grade = False


def main():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    bart_model = BartForDataToText.from_pretrained('facebook/bart-base')    
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/web_nlg_train.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/web_nlg_test.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/web_nlg_dev.csv'], batch_size = 1)

    summary_data.prepare_data()
    hparams = argparse.Namespace()

    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4
    model = LitModel(learning_rate = 1e-5, tokenizer = tokenizer, model = bart_model, hparams = hparams)
    checkpoint = ModelCheckpoint('checkpoint_files_2/')
    trainer = pl.Trainer(gpus=2, accelerator='dp',
                        max_epochs = 100,
                        min_epochs = 1,
                        auto_lr_find = False,
                        checkpoint_callback = checkpoint,
                        progress_bar_refresh_rate = 100,
                        logger=logger)

    trainer.fit(model, summary_data)
    trainer.save_checkpoint("webnlg_sanity_model.ckpt")


def inference():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = LitModel.load_from_checkpoint(checkpoint_path="webnlg_sanity_model.ckpt")
    print("Model loaded")
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/web_nlg_train.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/web_nlg_test.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/web_nlg_dev.csv'], batch_size = 1)
    summary_data.prepare_data()
    summary_data.setup("stage")
    train_data = summary_data.train_dataloader()

    it = iter(train_data)
    for batch in it:
        model.generate_text(batch)

if __name__ == '__main__': 
    #main()
    inference()