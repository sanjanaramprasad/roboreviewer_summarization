import json 
import pandas as pd

def read_json(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

def parse(entry):
    print(entry)
    target = entry["lexicalisations"][0]["lex"]
    triplesets = entry["originaltriplesets"]["originaltripleset"][0]

    object_t = [each_obj["object"] for each_obj in triplesets]
    property_t = [each_obj["property"] for each_obj in triplesets]
    subject_t = [each_obj["subject"] for each_obj in triplesets]
    return object_t, property_t, subject_t, target
    

def parse_json(contents):
    entries = contents["entries"]
    sources = []
    targets = []
    for each_entry in entries:
        each_entry_keys = each_entry.keys()
        for key in each_entry_keys:
            col0, col1, col2, target  = parse(each_entry[key])
            data_entry = {  "col0": col0,
                            "col1": col1,
                            "col2": col2}
            sources.append(data_entry)
            targets.append(target)
    return sources, targets

def write_data_csv(src_data, tgt_data, filename):
    print(tgt_data[:5])
    df = pd.DataFrame({'source': src_data, 'target': tgt_data},
                        columns = ['source', 'target'])
    df.to_csv(filename)



train_file ="/Users/sanjana/roboreviewer_summarization/data/webnlg-dataset/release_v2/json/webnlg_release_v2_train.json"
train_contents = read_json(train_file)
source_contents_train, target_contents_train = parse_json(train_contents)
write_data_csv(source_contents_train, target_contents_train, "web_nlg_train.csv")

dev_file ="/Users/sanjana/roboreviewer_summarization/data/webnlg-dataset/release_v2/json/webnlg_release_v2_dev.json"
dev_contents = read_json(dev_file)
source_contents_dev, target_contents_dev = parse_json(dev_contents)
write_data_csv(source_contents_dev, target_contents_dev, "web_nlg_dev.csv")

test_file ="/Users/sanjana/roboreviewer_summarization/data/webnlg-dataset/release_v2/json/webnlg_release_v2_test.json"
test_contents = read_json(test_file)
source_contents_test, target_contents_test = parse_json(test_contents)
write_data_csv(source_contents_test, target_contents_test, "web_nlg_test.csv")


