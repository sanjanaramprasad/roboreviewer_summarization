import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
#import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper, BartModel
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper
import torch



def preprocess_df(df, keys):
    for key in keys:
        df = df[df[key] != "['']"]
    return df


def encode_sentences(tokenizer, df, surface_keys, content_keys, targets, max_length=1024, pad_to_max_length=True, return_tensors="pt"):

    
    encoded_sentences = {}

    target_ids = []
    
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
    
    for _, row in df.iterrows():
        content_val = []
        num_items = len(eval(row[content_keys[0]]))
        row_contents = {}
        for idx in range(0, num_items):
            idx_vals_content = ["<%s> "%k+ eval(row[k])[idx] + " </%s>"%k for k in content_keys]
            content_sent = " ".join(idx_vals_content)
            content_sent = "<study> " + content_sent + " </study>" 
            content_val.append(  content_sent )

        for surface_k in surface_keys:
            row_surface_k = eval(row[surface_k])
            row_surface_k = " ".join(["<study> <%s> "%surface_k+ each + " </%s> </study>"%surface_k for each in row_surface_k])
            row_contents[surface_k] =  row_surface_k.strip()

        row_content_sent = " ".join(content_val)
        row_contents['content'] = row_content_sent.strip()
        
        
        
        
        for key, key_sent in row_contents.items():
            id_key = '%s_ids'%key
            attention_mask_key = '%s_attention_masks'%key
            if id_key not in encoded_sentences:
                encoded_sentences[id_key] = []
                encoded_sentences[attention_mask_key] = []
                
            sentence_encoding = run_bart(key_sent.strip())
            
            encoded_sentences[id_key].append(sentence_encoding['input_ids'])
            encoded_sentences[attention_mask_key].append(sentence_encoding['attention_mask'])
            
    for tgt_sentence in targets:
        encoded_dict = tokenizer(
              tgt_sentence,
              max_length=max_length,
              padding="max_length" if pad_to_max_length else None,
              truncation=True,
              return_tensors=return_tensors,
              add_prefix_space = True
        )
        # Shift the target ids to the right
        #shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])
        
    for key in list(encoded_sentences.keys()):
        encoded_sentences[key] = torch.cat(encoded_sentences[key], dim = 0)
        
    target_ids = torch.cat(target_ids, dim = 0)
    encoded_sentences['labels'] = target_ids
        
        

    return encoded_sentences

    

def preprocess_df(df, keys):
    for key in keys:
        df = df[df[key] != "['']"]
    return df

class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_files, batch_size, num_examples = 20000 , max_len = 1024, flatten_studies = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_files = data_files
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.max_len = max_len
        self.flatten_studies = flatten_studies

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        self.train = pd.read_csv(self.data_files[0])
        self.validate = pd.read_csv(self.data_files[1])
        self.test = pd.read_csv(self.data_files[2])
        preprocess_keys = ['population', 'interventions', 'outcomes', 'SummaryConclusions','punchline_text', 'punchline_effect' ]
        self.train = preprocess_df(self.train, preprocess_keys)
        self.validate = preprocess_df(self.validate, preprocess_keys)
        self.test = preprocess_df(self.test, preprocess_keys)

    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, 
                                      self.train,
                                        ['population', 'punchline_text'],
                                        ['interventions_mesh', 'outcomes_mesh', 'punchline_effect'], 
                                        self.train['SummaryConclusions'],
                                        max_length = self.max_len)
        
        self.validate = encode_sentences(self.tokenizer, 
                                      self.validate,
                                        ['population', 'punchline_text'],
                                        ['interventions_mesh', 'outcomes_mesh', 'punchline_effect'], 
                                        self.validate['SummaryConclusions'],
                                        max_length = self.max_len)
        self.test = encode_sentences(self.tokenizer, 
                                      self.test,
                                        ['population', 'punchline_text'],
                                        ['interventions_mesh', 'outcomes_mesh', 'punchline_effect'], 
                                        self.test['SummaryConclusions'],
                                        max_length = self.max_len)
        
    def train_dataloader(self, data_type = 'robo'):
        #dataset = TensorDataset
        dataset = TensorDataset(self.train['population_ids'], self.train['population_attention_masks'],
                                self.train['punchline_text_ids'], self.train['punchline_text_attention_masks'],
                                self.train['content_ids'], self.train['content_attention_masks'],
                                    self.train['labels'])
        #dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self, data_type = 'robo'):
        dataset = TensorDataset(self.validate['population_ids'], self.validate['population_attention_masks'],
                                self.validate['punchline_text_ids'], self.validate['punchline_text_attention_masks'],
                                self.validate['content_ids'], self.validate['content_attention_masks'],
                                    self.validate['labels'])
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self, data_type = 'robo'):
        #print(self.test['punchline_text_ids'])
        dataset = TensorDataset(self.test['population_ids'], self.test['population_attention_masks'],
                                self.test['punchline_text_ids'], self.test['punchline_text_attention_masks'],
                                self.test['content_ids'], self.test['content_attention_masks'],
                                    self.train['labels'])
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data

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

if __name__ == '__main__':
    additional_special_tokens = ["<sep>", "<content>", "</content>",
            "<surface>", "</surface>",
            "<outcomes_mesh>", "</outcomes_mesh>",
            "<punchline_text>", "</punchline_text>",
            "<population>", "</population>",
            "<interventions_mesh>", "</interventions_mesh>",
            "<punchline_effect>", "</punchline_effect>"]
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>", 
                                                    eos_token="</s>", 
                                                    pad_token = "<pad>")

    tokenizer.add_tokens(additional_special_tokens)
    #bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')    
    data_files = ['train_rr_data.csv', 'dev_rr_data.csv' , 'test_rr_data.csv']

    
                                    
    
    summary_data = make_data(tokenizer, SummaryDataModule, data_type = 'robo', path = '/home/ramprasad.sa', files = data_files, max_len = 1024)
    print(summary_data.train)
    summary_data.setup("stage")
    it = summary_data.val_dataloader()
    batches = iter(it)
    batch = next(batches)

    def print_pico(batch):
        content_input_ids = batch[0] if len(batch) >1 else None
        content_attention_masks = batch[1] if len(batch) >1 else None
        print("CONTENT")
        print(" ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in content_input_ids]))
        print(content_attention_masks)
        print('=' * 13)

        population_input_ids = batch[2] if len(batch) >1 else None
        population_attention_masks = batch[3] if len(batch) >1 else None
        print("POPULATION")
        print(" ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in population_input_ids]))
        print(population_attention_masks)
        print('=' * 13)

        punchline_text_input_ids = batch[4] if len(batch) >1 else None
        punchline_text_attention_masks = batch[5] if len(batch) >1 else None
        print("PUNCHLINE_TEXT")
        print(" ".join([tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in punchline_text_input_ids]))
        print(punchline_text_attention_masks)
        print('=' * 13)

    print_pico(batch)


        
