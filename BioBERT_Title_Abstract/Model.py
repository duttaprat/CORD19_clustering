import os
import json
import random
import numpy as np
import pandas as pd
import transformers
import warnings
import torch
import gc
import nltk
import re
nltk.download('punkt')
covid_df = pd.read_pickle("Covid19_title_abstract.pkl")

from transformers import AutoTokenizer, AutoModel, AutoConfig
config = AutoConfig.from_pretrained("https://s3.amazonaws.com/models.huggingface.co/bert/monologg/biobert_v1.1_pubmed/config.json",output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model = AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed",config=config)

numbers=[str(item) for item in range(1900, 2021)]
numbers2=[str(item)+";" for item in range(1900, 2021)]
numbers.append(numbers2)
stopwords=['introduction','background','abstract','title:','introduction:','background:','abstract:','title','doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved','conclusion', 'permission', 'used','al.' 'using', 'biorxiv', 'medrxiv', 'license', 'Fig', 'Fig.','Figs','Figs.','fig', 'fig.', 'et','al','day',
    'al.', 'elsevier', 'PMC', 'CZI', 'www','com','abstract:','introduction','that','because','of','title','abstract','cite',"Wang",
    "Devi","Zhang","Li","Liu","Singh","Yang","Kumar","Wu","Xu","Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson", "Anderson", "Taylor"]
from nltk import sent_tokenize
for i, row in covid_df.iterrows():
  clean = re.sub(r"""
               [,@#;:?!()&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               row['all_text'], flags=re.VERBOSE)
  words=clean.split()
  resultwords= [ word for word in words if word.lower() not in stopwords and word not in numbers]
  text = ' '.join([i for i in resultwords])
  
  tokenized = tokenizer.encode(text, add_special_tokens=True,truncation=True)
  segments_ids = [1] * len(tokenized)
  input_ids = torch.tensor([tokenized])
  segment_ids = torch.tensor([segments_ids])

  with torch.no_grad():
    outputs = model(input_ids, segment_ids)
    hidden_states = outputs[2]
  token_vecs = hidden_states[-2][0]
  if i==0 :
      features = torch.mean(token_vecs, dim=0).numpy()
      features=np.reshape(features,[1,768])
      print(features.shape)
  else:
      feature = torch.mean(token_vecs, dim=0).numpy()
      feature=np.reshape(feature,[1,768])
      features=np.append(features,feature,axis=0)
      print(features.shape)
np.save("Biobert_title_abstract.npy",features)
