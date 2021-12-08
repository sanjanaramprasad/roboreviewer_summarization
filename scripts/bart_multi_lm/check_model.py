from DataToTextProcessor import SummaryDataModule
#from Data2TextProcessor_1 import SummaryDataModule
from transformers import BartTokenizer
import torch.optim as optim



from torch import nn 
import torch


def make_data(tokenizer, SummaryDataModule,  data_type = 'robo', path = '/home/ramprasad.sa', files = ['robo_train_sep.csv', 'robo_dev_sep.csv', 'robo_test_sep.csv']):
    if data_type == 'robo':
        train_file = path + '/summarization/datasets/%s'%(files[0])
        dev_file = path + '/summarization/datasets/%s'%(files[1])
        test_file = path + '/summarization/datasets/%s'%(files[2])

    data_files = [train_file, dev_file, test_file]
    summary_data = SummaryDataModule(tokenizer, data_files = data_files,  batch_size = 3, max_len = 1024)
    summary_data.prepare_data()
    return summary_data


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

#additional_special_tokens = ["<sep>"]
additional_special_tokens = ['<population>', '</population>',
                                        '<interventions>', '</interventions>',
                                        '<outcomes>', '</outcomes>',
                                        '<punchline_text>', '</punchline_text>',
                                        '<punchline_effect>', '</punchline_effect>', "<sep>", "<bos>"]
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', bos_token="<s>",
                                                    eos_token="</s>",
                                                    pad_token = "<pad>")

tokenizer.add_tokens(additional_special_tokens)


class BartMultiEncHATTester():

    def test_model_forward_bart_encoder(self, encoder_combination_type):
        from model import BartForDataToTextGeneration_MultiLM
        
        self.model = BartForDataToTextGeneration_MultiLM.from_pretrained('facebook/bart-base')
        self.model.resize_token_embeddings(len(tokenizer))
        self.model._make_multiple_lm_heads()
        print("Loading Data ...")
        summary_data = make_data(tokenizer, SummaryDataModule, path = '/home/ramprasad.sa', files = ['train_rr_data.csv', 
                            'dev_rr_data.csv', 'test_rr_data.csv'])
        summary_data.setup()
        val_data = summary_data.val_dataloader(data_type = 'robo')
        print("Done.")
        it = iter(val_data)

        data = next(it)
        population_input_ids, population_attention_masks,\
                interventions_input_ids, interventions_attention_masks,\
                outcomes_input_ids, outcomes_attention_masks,\
                punchline_text_input_ids, punchline_text_attention_masks, = get_data(data)

        print("forward...") 
        tgt_ids = data[-1]
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

        #tgt_ids = data[-1]
        optimizer = optim.Adam(self.model.parameters())
        loss = outputs[0]
        #ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        #loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        print("OUTPUTS", outputs[0])
        print('=' *13)

    
        

obj = BartMultiEncHATTester()
#obj.test_model_forward_bart_encoder(encoder_combination_type = 'HAT')
obj.test_model_forward_bart_encoder(encoder_combination_type='self_attention')
#obj.test_model_forward_bart_encoder_loop_per_study(encoder_combination_type = 'linearize')
