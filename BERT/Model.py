import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd
import scipy
import re
import logging
import random
import numpy as np
import pandas as pd
import transformers
import warnings
import torch
import nltk
nltk.download('punkt')

# Using Biobert v1.1
covid_df = pd.read_pickle("CORD19.pkl")
from transformers import BertTokenizer, BertModel,BertConfig
config = BertConfig.from_pretrained('bert-base-uncased',output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',config=config)

device= torch.device("cuda:0")
model.to(device)
model.eval()

#Stopwords removal
numbers=[str(item) for item in range(1900, 2021)]
numbers2=[str(item)+";" for item in range(1900, 2021)]
numbers.append(numbers2)
stopwords=['introduction','document','begin','end','amssymb','documentclass','wasysym','oddsidemargin','12pt','69pt','minimal','mathrsfs','amsbsy','background','abstract','title:','introduction:','background:','abstract:','title','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved','conclusion', 'permission', 'used','al.', 'using', 'biorxiv', 'medrxiv', 'license', 'Fig', 'Fig.','Figs','Figs.','fig', 'fig.', 'et','al','day',
    'al.', 'elsevier', 'PMC', 'CZI', 'www','com','abstract:','introduction','that','because','of','title','abstract','cite',"Wang","may",
    "Devi","Zhang","Li","Abstract","Liu","Singh","Yang","Kumar","Wu","Xu","Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson", "Anderson", "Taylor","usepackage","amsfonts","amsmath","amsby","setlength","upgreek"]
from nltk import sent_tokenize

for i, row in covid_df.iterrows():
# Punctuation removal
  clean = re.sub(r"""
               [,@#;:?!()&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               row['all_text'], flags=re.VERBOSE)
  words=clean.split()
  resultwords= [word for word in words if word.lower() not in stopwords and word not in numbers]
  text = ' '.join([i for i in resultwords])
  # Create sentence tokens
  ls = sent_tokenize(text)
  l=[s for s in ls if len(s)>=50]
  tokenized = [tokenizer.encode(x, add_special_tokens=True,truncation=True) for x in l]
  if len(tokenized)>110:
      tokenized=tokenized[:110]
  max_len = 0
  for x in tokenized:
    if len(x) > max_len:
        max_len = len(x)
  padded = np.array([x + [0]*(max_len-len(x)) for x in tokenized])
  attention_mask = np.where(padded != 0, 1, 0)
  input_ids = torch.tensor(padded).to(device)
  attention_mask = torch.tensor(attention_mask).to(device)
  with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    hidden_states = outputs[2]
  token_vecs = hidden_states[-2]
  if i==0 :
      features = torch.mean(token_vecs, dim=1).numpy()
      if features.shape[0]!=110:
        features= np.pad(features, ((0, 110 - features.shape[0]),(0,0)), 'constant')
      features1=np.reshape(features,[1,768*110])
      print("Features Dimensions for i = {} : {}".format(i,features.shape))
      print("Features1 Dimensions for i = {} : {}".format(i,features1.shape))
  else:
      feature = torch.mean(token_vecs, dim=1).numpy()
      if feature.shape[0]!=110:
        feature= np.pad(feature, ((0, 110 - feature.shape[0]),(0,0)), 'constant')
      features=np.dstack((features,feature))
      feature1=np.reshape(feature,[1,768*110])
      features1=np.dstack((features1,feature1))
      print("Features Dimensions for i = {} : {}".format(i,features.shape))
      print("Features1 Dimensions for i = {} : {}".format(i,features1.shape))

features1=np.reshape(features1,[features1.shape[0],768*110])
np.save("Bert_vectors.npy",features1)
