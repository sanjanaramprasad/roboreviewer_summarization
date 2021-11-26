from transformers import BartTokenizer
from model import BartForDataToTextGeneration_MultiLM
from DataToTextProcessor import SummaryDataModule
import subprocess, os, sys 
from run_experiment import LitModel



def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/ramprasad.sa', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv']):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 3, max_len = 1024)
    summary_data.prepare_data()
    return summary_data

'''additional_special_tokens = ["<sep>", "<study>", "</study>",
            "<outcomes>", "</outcomes>",
            "<punchline_text>", "</punchline_text>",
            "<population>", "</population>",
            "<interventions>", "</interventions>",
            "<punchline_effect>", "</punchline_effect>"]
'''    
#tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",eos_token="</s>",pad_token = "<pad>")

    
#tokenizer.add_tokens(additional_special_tokens)

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
freeze_encoder = True
freeze_embeds = True
eval_beams = 3

#data_files = ['train_rr_data.csv', 'dev_rr_data.csv' , 'test_rr_data.csv']
#summary_data = make_data(tokenizer, SummaryDataModule, data_type = 'robo', path = '/home/ramprasad.sa', files = data_files, max_len = 1024)

learning_rate = 3e-5
checkpoint_file = 'checkpoint_files_final/token_mixture_lm_background/epoch=4-val_loss=0.27.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_file, learning_rate = learning_rate, model=bart_model, tokenizer = tokenizer,  freeze_encoder = freeze_encoder, freeze_embeds = freeze_embeds, eval_beams = eval_beams)

num_beams = 3
min_len = 90
repetition_penalty = 1.0
length_penalty = 2.0
    
