
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper

from BartForDataToTextGeneration import BartForDataToText
from Data2TextProcessor_1 import SummaryDataModule


model = BartForDataToText.from_pretrained('facebook/bart-base')    
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
summary_data = SummaryDataModule(tokenizer, data_files = ['/home/sanjana/roboreviewer_summarization/data/web_nlg_train.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/web_nlg_dev.csv', 
                                           '/home/sanjana/roboreviewer_summarization/data/web_nlg_test.csv'], batch_size = 1)
summary_data.prepare_data()

summary_data.setup("stage")
test_data = summary_data.test_dataloader(data_type = 'webnlg')
it = iter(test_data)

class BartForDataToTextGenerationTester():

    def test_get_encoders(self):
        encoder_col0, encoder_col1, \
            encoder_col2, encoder_col3, encoder_col4 = model.get_encoders() 

        self.encoder_col0 = encoder_col0
        self.encoder_col1 = encoder_col1 
        self.encoder_col2 = encoder_col2 
        self.encoder_col3 = encoder_col3 
        self.encoder_col4 = encoder_col4

        return

    def test_encoder0(self):
        data = next(it)
        print(len(data))
        encoder_outputs_col0 = self.encoder_col0(\
                                input_ids = data[0],
                                attention_mask = data[1])
        print(encoder_outputs_col0[0].shape)
        return
        

    
        
obj = BartForDataToTextGenerationTester()
obj.test_get_encoders()
obj.test_encoder0()
    