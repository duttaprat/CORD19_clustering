import os
import json
import re
import sys
import gensim
import numpy as np
import pandas as pd
import nltk
import logging
from gensim.models import Doc2Vec
from gensim.parsing.porter import PorterStemmer
from gensim.parsing.preprocessing import remove_stopwords
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords=stopwords.words('english')
stemmer = SnowballStemmer(language="english")

custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www','com','abstract','introduction','that','because','of','title','abstract','cite'
]
corpus=[]
num=0
# Building a corpus after removal of unnecessary punctuationsand stopwords, and stemming.
for i, row in covid_df.iterrows():
    num+=1
    w = re.sub("\'", "", row['all_text'])
    doc = remove_stopwords(w)
    tokens = gensim.utils.simple_preprocess(doc,  deacc=True)
    tokens = [stemmer.stem(token) for token in tokens]
    mytokens = [ word for word in tokens if word not in custom_stop_words]
    mytokens = [ word for word in mytokens if word not in stopwords]
    corpus.append(gensim.models.doc2vec.TaggedDocument(words=mytokens,tags=[str(num)]))

"""###**Model 1: Doc2vec PV-DM (based on CBOW in word2vec)**
1.   Model 1.1: Taking mean
2.   Model 1.2: Taking sum of context words

**Model 1.1: Notable Results found with only this model henceforth.**

"""

model1_dm =gensim.models.doc2vec.Doc2Vec(vector_size=300, epochs=100,dm=1,alpha=0.025,min_alpha =0.0001, window=5,dm_mean=1,negative=5,workers=5)
model1_dm.build_vocab(corpus,min_count=20)
model1_dm.train(corpus, total_examples=model1_dm.corpus_count, epochs=model1_dm.epochs)
model1_dm.save("doc2vec_models/model1_dm_300")

"""**Model 1.2**"""

model2_dm =gensim.models.doc2vec.Doc2Vec(vector_size=300, epochs=100,dm=0,alpha=0.025,min_alpha =0.0001, window=5,dm_mean=0,negative=5,workers=5)
model2_dm.build_vocab(corpus,min_count=20)
model2_dm.train(corpus, total_examples=model2_dm.corpus_count, epochs=model2_dm.epochs)
model2_dm.save("doc2vec_models/model2_dm_300")

"""###**Model 2: Doc2vec PV-DBOW (based on skipgrams in word2vec)**
1.   Model 2.1: With negative sampling
2.   Model 2.2: With hierarchal softmax
"""

model1_dbow =gensim.models.doc2vec.Doc2Vec(vector_size=300, epochs=40,dm=0,alpha=0.025,min_alpha =0.0001,window=15, negative=5,workers=5)
model1_dbow.build_vocab(corpus,min_count=20)
model1_dbow.train(corpus, total_examples=model1_dbow.corpus_count, epochs=model1_dbow.epochs)
model1_dbow.save("doc2vec_models/model1_dbow_300")

model2_dbow =gensim.models.doc2vec.Doc2Vec(vector_size=300, epochs=40,dm=0,alpha=0.025,min_alpha =0.0001,window=15,hs=1,workers=5)
model2_dbow.build_vocab(corpus,min_count=20)
model2_dbow.train(corpus, total_examples=model2_dbow.corpus_count, epochs=model2_dbow.epochs)
model2_dbow.save("doc2vec_models/model2_dbow_300")
