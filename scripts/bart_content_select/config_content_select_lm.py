from transformers import BartTokenizer
from Data2TextProcessor import SummaryDataModule
import subprocess, os, sys 
from run_experiment_multiLM import LitModel

def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/Users/sanjana', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv'], max_len = 256):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    print(train_file)
    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 1, max_len = max_len, flatten_studies = True)
    summary_data.prepare_data()
    
    assert(len(summary_data.train) > 10)
    return summary_data


additional_special_tokens = ["<sep>", "<study>", "</study>",
            "<outcomes_mesh>", "</outcomes_mesh>",
            "<punchline_text>", "</punchline_text>",
            "<population_mesh>", "</population_mesh>",
            "<interventions_mesh>", "</interventions_mesh>",
            "<punchline_effect>", "</punchline_effect>"]
    
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",
                                                    eos_token="</s>",
                                                    pad_token = "<pad>")

    
tokenizer.add_tokens(additional_special_tokens)
data_files = ['train_rr_data.csv', 'dev_rr_data.csv' , 'test_rr_data.csv']
summary_data = make_data(tokenizer, SummaryDataModule, data_type = 'robo', path = '/home/ramprasad.sa', files = data_files, max_len = 1024)


checkpoint_file = 'checkpoint_files_final/token_mixture_lm/epoch=3-val_loss=0.27.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_file)

num_beams = 4
min_len = 90
repetition_penalty = 1.0
length_penalty = 2.0
    
