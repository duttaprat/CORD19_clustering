
import sklearn
import os
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import logging
import numpy as np
import random
import sys
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

vectors = np.load("Biobert_vectors.npy")
vectors = scaler.fit_transform(vectors)
nc = [5,6,7]
AC = [AgglomerativeClustering(n_clusters=i, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None) for i in nc]

def Hierarchal(AC,nc,vectors):  
    for i in range(0,len(nc)):  
        labels = AC[i].fit_predict(vectors)
        db= sklearn.metrics.silhouette_score(vectors,labels)
        score_ss=sklearn.metrics.silhouette_score(vectors, labels, metric='euclidean')
        score_db=sklearn.metrics.davies_bouldin_score(vectors, labels)
        score_ch = sklearn.metrics.calinski_harabasz_score(vectors, labels)
        logging.warning("SS for {} = {}".format(nc[i],score_ss))
        logging.warning("DB for {} = {}".format(nc[i],score_db))
        logging.warning("CH for {} = {}".format(nc[i],score_ch))
Hierarchal(AC,nc,vectors)
