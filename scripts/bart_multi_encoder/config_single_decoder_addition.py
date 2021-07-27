#from transformers import BartTokenizer
from Data2TextProcessor import SummaryDataModule
from transformers import BartTokenizer
import subprocess, os, sys 
from run_experiment_decoder import LitModel


parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir_name = parent_dir_name.split('scripts')[0].strip()

def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv']):
    #parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #print(parent_dir_name)
    if data_type == 'robo':
        train_file = parent_dir_name + 'data/bart_multienc_per_key/%s'%(files[0])
        dev_file = parent_dir_name + 'data/bart_multienc_per_key/%s'%(files[1])
        test_file = parent_dir_name + 'data/bart_multienc_per_key/%s'%(files[2])

    elif data_type =='webnlg':
        train_file = parent_dir_name + '/roboreviewer_summarization/data/web_nlg_train.csv'
        dev_file = parent_dir_name + '/roboreviewer_summarization/data/web_nlg_dev.csv'
        test_file = parent_dir_name + '/roboreviewer_summarization/data/web_nlg_test.csv'

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1)
    summary_data.prepare_data()
    return summary_data


additional_special_tokens = ["<sep>", "<study>", "</study>",
            "<outcomes>", "</outcomes>",
            "<punchline_text>", "</punchline_text>",
            "<population>", "</population>",
            "<interventions>", "</interventions>",
            "<punchline_effect>", "</punchline_effect>"]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",
                                                    eos_token="</s>",
                                                    pad_token = "<pad>")


summary_data = make_data(tokenizer, SummaryDataModule, data_type = 'robo')

checkpoint_file = 'scripts/bart_multi_encoder/checkpoint_files_final/3e-5_decomod_addition/epoch=4-val_loss=0.25.ckpt'
print(parent_dir_name + checkpoint_file)
model = LitModel.load_from_checkpoint(checkpoint_path=parent_dir_name + checkpoint_file)

num_beams = 3
min_len = 90
repetition_penalty = 2.0
length_penalty = 2.0
    
