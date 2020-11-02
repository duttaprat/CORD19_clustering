import sklearn
import os
from sklearn import metrics
import skfuzzy
import logging
import numpy as np
import random
import sys
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
## Notable Results with Model1_dm
vectors = np.load("/../model1_dm_300.docvecs.vectors_docs.npy")
vectors = np.reshape(vectors,[300,48585])
vectors = scaler.fit_transform(vectors)
nc = [2,3,4,5,6,7,8,10,15,20,40,60,80,120,160,200]
def FCM(nc,vectors):  
    for i in range(0,len(nc)):  
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cmeans(vectors, nc[i], 2, error=0.005, maxiter=500, init=None, seed=None)
        labels = np.argmax(u, axis=0)
        logging.warning(labels.shape)
        vectors = np.reshape(vectors,[48585,300])
        db= sklearn.metrics.silhouette_score(vectors,labels)
        score_ss=sklearn.metrics.silhouette_score(vectors, labels, metric='euclidean')
        score_db=sklearn.metrics.davies_bouldin_score(vectors, labels)
        score_ch = sklearn.metrics.calinski_harabasz_score(vectors, labels)
        logging.warning("SS for {} = {}".format(nc[i],score_ss))
        logging.warning("DB for {} = {}".format(nc[i],score_db))
        logging.warning("CH for {} = {}".format(nc[i],score_ch))
        logging.warning("FPC for {} = {}".format(nc[i],fpc))
        vectors = np.reshape(vectors,[300,48585])
FCM(nc,vectors)
