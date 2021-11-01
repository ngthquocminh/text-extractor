#!/usr/bin/env python
# coding: utf-8

# In[3]:


traindata_file = "traindata_1.json"
model="vi_core_news_lg"
new_model_name="training1"
output_dir='/output'
n_iter=20


# In[4]:


import json
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import random


# In[5]:


def load_data(file):
    with open(traindata_file, 'r', encoding="utf8") as f:
        return [json.loads(line) for line in f.readlines()]


# In[6]:


data = load_data(traindata_file)
data


# In[7]:



def convert_data(traindata_file):
    try:
        training_data = []
        lines = load_data(traindata_file)
        for line in lines:
            data = line

            text = data['content']
            entities = []
            if data['annotation'] is not None:
                for annotation in data['annotation']:
                    # only a single point in text annotation.
                    point = annotation['points'][0]
                    labels = annotation['label']
                    # handle both list of labels or a single label.
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        # dataturks indices are both inclusive [start, end]
                        # but spacy is not [start, end)
                        entities.append((
                            point['start'],
                            point['end'], # + 1,
                            label
                        ))

            training_data.append((text, {"entities": entities}))
        return training_data
    except Exception:
#         logging.exception("Unable to process " + traindata_file)
        return None

SPACY_DATA = convert_data(traindata_file)
print(SPACY_DATA)


# In[ ]:



from spacy.training import Example
from spacy.util import minibatch

nlp = spacy.blank("vi")

if "ner" not in nlp.pipe_names:
    nlp.add_pipe("ner", last=True)
    
examples = []
for text, annots in SPACY_DATA:
    examples.append(Example.from_dict(nlp.make_doc(text), annots))
    
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

with nlp.disable_pipes(*other_pipes):
    nlp.initialize(lambda: examples)
    for i in range(20):
        random.shuffle(examples)
        for batch in minibatch(examples, size=8):
            nlp.update(batch)
        


# In[18]:


# nlp.to_disk("./en_example_pipeline")
doc = nlp('Loại tin Bán nhà riêng Tên liên hệ: Phan Thanh Tùng Giá 4,7 Tỷ Điện tích 38 m² Số tầng 5 Số phòng 0 hôm nay 0912142902 Mặt tiền 0 m')
for ent in doc.ents:
    print(" > ", ent.label_, ent.text)


# In[ ]:


import random

def train_spacy(TRAIN_DATA, iterations):

    #Create the blank spacy model
    nlp = spacy.blank("en")
    
    #add the ner component to the pipeline if it's not there
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    
    #add all labels to the spaCy model
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    #eliminate the effect of the training on other pipes and 
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    
    #begin training
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print ("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                            [text],
                            [annotations],
                            drop=0.2,
                            sgd=optimizer,
                            losses=losses
                )
            print
    return (nlp)

#run function and create a trained model
trained_nlp = train_spacy(TRAIN_DATA, 10)


# In[72]:


nlp = spacy.blank("vi")
def create_spacy_data(data): 
    docbin = DocBin()
    for text, annot in tqdm(data):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span:
                ents.append(span)
            else:
                print("skip")
        doc.ents = ents
        docbin.add(doc)
    return docbin
split_point = int(len(SPACY_DATA)*0.9)
train_data = create_spacy_data(SPACY_DATA[0:split_point])
train_data.to_disk("./data/train_data.spacy")
valid_data = create_spacy_data(SPACY_DATA[split_point:-1])
valid_data.to_disk("./data/valid_data.spacy")


# In[ ]:




