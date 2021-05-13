
from transformers import BartTokenizer, BartForCausalLM, BartForConditionalGeneration, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, TopKLogitsWarper, TemperatureLogitsWarper

from BartForDataToTextGeneration import BartForDataToText
from Data2TextProcessor_1 import SummaryDataModule


model = BartForDataToText.from_pretrained('facebook/bart-base')    
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
summary_data = SummaryDataModule(tokenizer, data_files = ['/Users/sanjana/roboreviewer_summarization/data/web_nlg_train.csv', 
                                           '/Users/sanjana/roboreviewer_summarization/data/web_nlg_dev.csv', 
                                           '/Users/sanjana/roboreviewer_summarization/data/web_nlg_test.csv'], batch_size = 1)
summary_data.prepare_data()

summary_data.setup("stage")
test_data = summary_data.test_dataloader(data_type = 'webnlg')
it = iter(test_data)

class BartForDataToTextGenerationTester():

    def test_get_encoders(self):
        encoder_col0, encoder_col1, \
            encoder_col2, encoder_col3, encoder_col4 = model.get_encoders()

        data = next(it)
        encoder_outputs_col0 = encoder_col0(\
                                input_ids = data[0],
                                attention_mask = data[1])
        print(encoder_outputs_col0.shape)
        return
        

    
        

BartForDataToTextGenerationTester().test_get_encoders()
    