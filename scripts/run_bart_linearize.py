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
from Data2TextProcessor_linearize import SummaryDataModule
#from transformers.modeling_bart import shift_tokens_right
from rouge import Rouge

learning_rate = 3e-5 
max_epochs = 25

logger = TensorBoardLogger('tb_logs_final', name='my_model_epoch_bartconditional%s_%s'%(str(max_epochs), str(learning_rate)))


train_count = 0
val_count = 0

import os
import pytorch_lightning as pl



    
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
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.freeze_embeds_ = freeze_embeds
        #self.hparams = hparams
        #self.hparams.update(hparams)

        if self.freeze_encoder:
            freeze_params(self.model.get_encoder())


        if freeze_embeds:
            self.freeze_embeds()
        self.save_hyperparameters()
  
    def freeze_embeds(self):
        ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
        freeze_params(self.model.model.shared)
        for d in [self.model.model.encoder, self.model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)

    # Do a forward pass through the model
    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
  
    def configure_optimizers(self):
        print("PARAMS", self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr= self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        #print(batch)
    
        input_ids = batch[0] 
        attention_mask= batch[1]
        tgt_ids = batch[-1]
        outputs = self(
            input_ids= input_ids,
            attention_mask= attention_mask,
            labels = tgt_ids,
            decoder_input_ids = None,
            use_cache = False,
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
    
        
        input_ids = batch[0] 
        attention_mask= batch[1]
        tgt_ids = batch[-1]
        outputs = self(
            input_ids= input_ids,
            attention_mask= attention_mask,
            labels = tgt_ids,
            decoder_input_ids = None,
            use_cache = False,
        )
        
        lm_logits = outputs[1]
        # Create the loss function
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        # Calculate the loss on the un-shifted tokens
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("Train Loss", loss, self.current_epoch)

        epoch_dictionary={
            'val_loss': loss,
            'log': tensorboard_logs}
        return epoch_dictionary

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

        

def make_data(tokenizer, data_type = 'robo', path = '/home/sanjana'):
    if data_type == 'robo':
        train_file = path + '/roboreviewer_summarization/data/robo_train_linearized.csv'
        dev_file = path + '/roboreviewer_summarization/data/robo_dev_linearized.csv'
        test_file = path + '/roboreviewer_summarization/data/robo_test_linearized.csv'
    
    elif data_type =='webnlg':
        train_file = path + '/roboreviewer_summarization/data/web_nlg_train.csv'
        dev_file = path + '/roboreviewer_summarization/data/web_nlg_dev.csv'
        test_file = path + '/roboreviewer_summarization/data/web_nlg_test.csv'

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1)
    summary_data.prepare_data()
    return summary_data


def main():
    additional_special_tokens = ["<sep>", "<study>", "</study>", 
            "<outcomes>", "</outcomes>", 
            "<punchline_text>", "</punchline_text>", 
            "<population>", "</population>", 
            "<interventions>", "</interventions>", 
            "<punchline_effect>", "</punchline_effect>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', unk_token="<unk>",
                                                    bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")
    tokenizer.add_tokens(additional_special_tokens)
    bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    bart_model.resize_token_embeddings(len(tokenizer))
    
    summary_data = make_data(tokenizer, path = '/home/sanjana')

    #hparams = argparse.Namespace()
    freeze_encoder = True
    freeze_embeds = True
    eval_beams = 4

    model = LitModel(learning_rate = learning_rate, tokenizer = tokenizer, model = bart_model, freeze_encoder = freeze_encoder, freeze_embeds = freeze_embeds, eval_beams = eval_beams)
    checkpoint = ModelCheckpoint('checkpoint_files_final/3e-5_bartcond_linearized/',
                                filename = '{epoch}-{val_loss:.2f}',
                                save_top_k=13,
                                monitor = 'val_loss')
    trainer = pl.Trainer(gpus=2, accelerator='dp', 
			max_epochs = max_epochs,
                        min_epochs = 1,
                        auto_lr_find = False,
                        progress_bar_refresh_rate = 100,
                        logger=logger,
                        callbacks=[checkpoint])

    trainer.fit(model, summary_data)
    trainer.save_checkpoint("robo_model_epoch%s_adam_%s_bartconditional.ckpt"%(str(learning_rate), str(max_epochs)))

def inference(checkpoint_file):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    #model = LitModel.load_from_checkpoint(checkpoint_path="webnlg_model_14_sgd.ckpt")
    #bart_model = BartForDataToText.from_pretrained('facebook/bart-base')
    #bart_model._make_duplicate_encoders()
    hparams = argparse.Namespace()
    rouge = Rouge()
    hparams.freeze_encoder = True
    hparams.freeze_embeds = True
    hparams.eval_beams = 4
    model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_file)
    summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/robo_train_linearized.csv',
                                           '/home/sanjana/roboreviewer_summarization/data/robo_dev_linearized.csv',
                                           '/home/sanjana/roboreviewer_summarization/data/robo_test_linearized.csv'], batch_size = 1)
    summary_data.prepare_data()

    summary_data.setup("stage")
    val_data = summary_data.val_dataloader(data_type = 'robo')

    #num_val = len(list(val_data))
    num_val = 5
    print("NUM EXAMPLES", num_val)
    it = iter(val_data)
    ind = 0
    model_out = []
    references = []
    rouge = Rouge()
    '''while(ind < num_val):
        ind += 1
        text = next(it)'''
    for text in it:
        generated_ids = model.model.generate(
                text[0],
                attention_mask=text[1],
                use_cache=True,
                decoder_start_token_id = tokenizer.pad_token_id,
                num_beams= 4,
                max_length = 300,
                repetition_penalty = 1.1,
                early_stopping = True
        )
    
        model_output = " ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids])
        #print("="*13)
        #reference = data[ind]
        target = ' '.join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in text[-1]])
        references.append(target)
        model_out.append(model_output)
    print(rouge.get_scores(model_out, references, avg=True))
if __name__ == '__main__': 
    #main()
    inference('/home/sanjana/roboreviewer_summarization/epoch=6-val_loss=3.25.ckpt')
   
