"""
References:
1. @inproceedings{EREN2020,
	author = {Eren, E. Maksim. Solovyev, Nick. Nicholas, Charles. Raff, Edward. Johnson, Ben},
	title = {COVID-19 Kaggle Literature Organization},
	year = {2020},
	month = {September},
	location = {Virtual Event, CA, USA},
	note={Malware Research Group, University of Maryland Baltimore County. \url{https://github.com/MaksimEkin/COVID19-Literature-Clustering}},
    	doi = {10.1145/3395027.3419591},
    	howpublished = {ACM Symposium on Document Engineering 2020 (DocEng '20)}
}


2. https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv/comments#About-this-notebook
"""

import os
import json
import re
import sys
import numpy as np
import pandas as pd
import nltk
import random

#50,000 documents retrieval
covid1_dir = '../Nupur/COVID_document_clustering/document_parses/pdf_json/'
filenames1 = os.listdir(covid1_dir)
filenames1=random.sample(filenames1, 25000)
covid2_dir = '../Nupur/COVID_document_clustering/document_parses/pmc_json/'
filenames2 = os.listdir(covid2_dir)
filenames2=random.sample(filenames2, 25000)

#Pre-Processing from the 2 different sources in CORD-19 dataset
def creation(filenames,covid_dir):
  all_files = []
  for filename in filenames:
    filename = covid_dir + filename
    try: 
      file = json.load(open(filename, 'rb'))
    except ValueError:
      pass
    all_files.append(file)
   return all_files
all_files1 = creation(filenames1,covid1_dir)
all_files2 =creation(filenames2,covid2_dir)
def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    for section, text in texts:
        texts_di[section] += text
        body = ""
    for section, text in texts_di.items():
        body += section
        body += " "
        body += text
        body += " "
    return body
cleaned_files1 = []
for file in tqdm(all_files1):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_body(file['body_text']),
            format_body(file['abstract']),
        ]
        cleaned_files1.append(features)
col_names1 = [
    'paper_id', 
    'title',  
    'text', 
    'abstract'
]
clean1_df = pd.DataFrame(cleaned_files1, columns=col_names1)
cleaned_files2 = []
for file in tqdm(all_files2):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_body(file['body_text']),
        ]

        cleaned_files2.append(features)

col_names2 = [
    'paper_id', 
    'title', 
    'text', 
]
clean2_df = pd.DataFrame(cleaned_files2, columns=col_names2)

#Appending dataframes, Removal of duplicates and Null values
clean_df = clean1_df.append(clean2_df,ignore_index=True, sort=False)
clean_df.drop_duplicates(['title','text'],inplace=True)
clean_df['abstract']= clean_df['abstract'].fillna('abstract')
clean_df.reset_index(inplace=True)

#Retaining only English Documents
from tqdm import tqdm
import langdetect
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0
languages = []
for ii in tqdm(range(0,len(clean_df))):
    text = clean_df.iloc[ii]['text'].split(" ")
    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        except Exception as e:
                lang = "unknown"
                pass
    languages.append(lang)
languages_dict = {}
for lang in set(languages):
    languages_dict[lang] = languages.count(lang)

clean_df['language'] = languages
clean_df = clean_df[clean_df['language'] == 'en'] 
clean_df.info()
clean_df["all_text"] = clean_df["title"]+ clean_df["abstract"] + clean_df["text"]
selected_columns = clean_df[["paper_id","all_text"]]
covid_df = selected_columns.copy()
for index, row in covid_df.iterrows(): 
  row['paper_id']='text '+str(index+1)
covid_df.dropna(inplace=True)
covid_df.reset_index(drop=True,inplace=True)
covid_df.to_pickle("Cord19.pkl")
