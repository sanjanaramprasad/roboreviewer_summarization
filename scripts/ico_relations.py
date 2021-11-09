import json, requests
import pandas as pd
def read_file(filename):
        with open(filename, 'r') as fp:
                all_lines = fp.readlines()
        return all_lines


'''{'text': 'there is insufficient evidence to support or refute the use of any specific type of psychosocial intervention for the treatment of patients with acute and chronic mental illnesses. further research is required to determine whether this approach is superior to other types of therapies in terms of patient satisfaction and quality of life. future research should focus on the long-term effects of these interventions on people with severe and chronic symptoms of social anxiety and their carers. such studies should be adequately powered to detect clinically relevant differences between different types of treatments and include a standardised outcome measure.', 'pmid': '0', 'title': 'untitled', 'p': [{'text': 'patients with acute and chronic mental illnesses.', 'offsets': [131, 180], 'mesh': ['Patient', 'Mental Disorders'], 'duis': ['D010361', 'D001523'], 'cuis': ['C0030705', 'C0004936']}, {'text': 'people with severe and chronic symptoms of social anxiety and', 'offsets': [419, 480], 'mesh': ['symptoms', 'Social Anxiety'], 'duis': ['Q000175', 'D001007'], 'cuis': ['C0683368', 'C0424166']}], 'i': [{'text': 'psychosocial intervention', 'offsets': [84, 109], 'mesh': [], 'duis': [], 'cuis': []}], 'o': [{'text': 'patient satisfaction and quality of life.', 'offsets': [297, 338], 'mesh': ['Patient Satisfaction', 'Quality of Life'], 'duis': ['D017060', 'D011788'], 'cuis': ['C0030702', 'C0034380']}], 'frames': [{'i': {'text': 'psychosocial intervention', 'mesh': [], 'duis': [], 'prob': 0.99621284}, 'c': {'text': 'psychosocial intervention', 'mesh': [], 'duis': [], 'prob': 0.0027066234}, 'o': {'text': 'patient satisfaction and quality of life.', 'mesh': ['Patient Satisfaction', 'Quality of Life'], 'duis': ['D017060', 'D011788']}, 'label': 1, 'ev': {'text': 'further research is required to determine whether this approach is superior to other types of therapies in terms of patient satisfaction and quality of life.', 'prob': 0.032260906}}]}'''

def process_result(result):
        frames = result['frames']
        label_map = {1: 'sig_increase', -1: 'sig_decrease', 0:'no_diff'}
        if frames:
                #print(len(frames))
                frames_data = []
                for frame in frames:
                        icos = frame
                        #print(icos)
                        i_mesh = icos['i']['text'].split(' ')
                        c_mesh = icos['c']['text'].split(' ')
                        o_mesh = icos['o']['text'].split(' ')
                        label = frame['label']
                        #frame_data = {'i' : i_mesh, 'c': c_mesh, 'o': o_mesh, 'label': label}
                        frame_data = icos
                        if frame_data not in frames_data:
                                frames_data.append(frame_data)
                return frames_data
        return None
data = "there is insufficient evidence to support or refute the use of any specific type of psychosocial intervention for the treatment of patients with acute and chronic mental illnesses. further research is required to determine whether this approach is superior to other types of therapies in terms of patient satisfaction and quality of life. future research should focus on the long-term effects of these interventions on people with severe and chronic symptoms of social anxiety and their carers. such studies should be adequately powered to detect clinically relevant differences between different types of treatments and include a standardised outcome measure."

data1 = "caring for people in acute day hospitals is as effective as inpatient care in treating acutely ill psychiatric patients. however further data are still needed on the cost effectiveness of day hospitals." 
X = 0.01

def get_file_relations(filename):
        data = read_file(filename)
        docs = [{"text": data.strip()} for data in data]
        docs = {'docs': docs, 'params': {'ev_thresh' : X}}
        ret = requests.post(url = 'http://trialstreamer.ccs.neu.edu:8000/run_pipeline', data = json.dumps(docs))
        return_val = ret.json()
        data_relations = []
        for each in return_val:
                data_relations.append(process_result(each))
        return data_relations


##bart_avg_output = read_file('bart_average_outputs.txt')
#for data in bart_avg_output:
##docs = [{"text": data.strip()} for data in bart_avg_output]
##print(json.dumps(docs))
##docs = {'docs': docs, 'params': {'ev_thresh' : X}}
#docs = {'docs' : [{"text" : data}, {"text":data1}], 'params': { 'ev_thresh': X }}

#ret = requests.post(url = 'http://trialstreamer.ccs.neu.edu:8000/run_pipeline', data = json.dumps(docs))
#return_val = ret.json()
#print(return_val)

model_output_file = "/home/ramprasad.sa/roboreviewer_summarization/scripts/trial_output.txt"
reference_file = '/home/ramprasad.sa/roboreviewer_summarization/scripts/trial_ref.txt'


data_relations = get_file_relations(model_output_file)
##print(data_relations)
data_relations_targets = get_file_relations(reference_file)

model_outputs = read_file(model_output_file)
targets = read_file(reference_file)

dataf = {'model_outputs' : model_outputs, 'targets': targets, 'model_output_relations' : data_relations, 'target_relations': data_relations_targets}

df = pd.DataFrame(dataf)
print(df)
df.to_csv('bart_context_lm_relations.csv')
