
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

        self.encoder_outputs_list = []
        self.attn_list = []
        return

    def test_encoder0(self):
        data = next(it)
        print(len(data))
        encoder_outputs_col0 = self.encoder_col0(\
                                input_ids = data[0],
                                attention_mask = data[1])
        print(encoder_outputs_col0[0].shape)
        self.encoder_outputs_list.append(encoder_outputs_col0)
        self.attn_list.append(data[1])
        return

    def test_encoder1(self):
        data = next(it)
        print(len(data))
        if len(data) > 3:
            encoder_outputs_col1 = self.encoder_col1(\
                                    input_ids = data[2],
                                    attention_mask = data[3])
            print(encoder_outputs_col1[0].shape)
            self.encoder_outputs_list.append(encoder_outputs_col1)
            self.attn_list.append(data[3])
        return

    def test_encoder2(self):
        data = next(it)
        print(len(data))
        if len(data) > 5:
            encoder_outputs_col2 = self.encoder_col2(\
                                    input_ids = data[4],
                                    attention_mask = data[5])
            print(encoder_outputs_col2[0].shape)
            self.encoder_outputs_list.append(encoder_outputs_col2)
            self.attn_list.append(data[5])
        return

    def test_encoder3(self):
        data = next(it)
        print(len(data))
        if len(data) > 7:
            encoder_outputs_col3 = self.encoder_col3(\
                                    input_ids = data[6],
                                    attention_mask = data[7])
            print(encoder_outputs_col3[0].shape)
            self.encoder_outputs_list.append(encoder_outputs_col3)
            self.attn_list.append(data[7])
        return

    def test_encoder4(self):
        data = next(it)
        print(len(data))
        if len(data) > 9:
            encoder_outputs_col4 = self.encoder_col4(\
                                    input_ids = data[8],
                                    attention_mask = data[9])
            print(encoder_outputs_col4[0].shape)
            self.encoder_outputs_list.append(encoder_outputs_col4)
            self.attn_list.append(data[9])
        return

    def test_encoder_addition(self):
        encoder_outputs_added =  model._get_added_encoder_outputs(self.encoder_outputs_list)
        print(encoder_outputs_added[0].shape)
     
    def test_attn_masks_OR(self):
        attention_added = model._get_attention_masks_OR(self.attn_list)
        print(attention_added.shape)

    def test_model_forward(self):
        data = next(it)
        input_ids_col0 = data[0] if len(data) >1 else None
        attention_mask_col0 = data[1] if len(data) >1 else None

        input_ids_col1 = data[2] if len(data) >3 else None
        attention_mask_col1 = data[3] if len(data) >3 else None

        input_ids_col2 = data[4] if len(data) >5 else None
        attention_mask_col2 = data[5] if len(data) >5 else None

        input_ids_col3 = data[6] if len(data) >7 else None
        attention_mask_col3 = data[7] if len(data) >7 else None

        input_ids_col4 = data[8] if len(data) >9 else None
        attention_mask_col4 = data[9] if len(data) >9 else None

        
        outputs = model(
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
            labels = data[10]
        )

        print(outputs)


    

    
        
obj = BartForDataToTextGenerationTester()
obj.test_get_encoders()
obj.test_encoder0()
obj.test_encoder1()
obj.test_encoder2()
obj.test_encoder3()
obj.test_encoder4()
obj.test_encoder_addition()
obj.test_attn_masks_OR()
obj.test_model_forward()
    