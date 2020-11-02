import sklearn
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import numpy as np
import random
import sys
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
vectors=np.load("Biobert_vectors.npy")
vectors = scaler.fit_transform(vectors)
nc = [5,10,15,20,40]
kmeans = []
kmeans = [MiniBatchKMeans(n_clusters = i, n_init = 100, max_iter = 1000,batch_size= 5000) for i in nc]

def Kmeans(kmeans,nc,vectors):  
    for i in range(0,len(nc)):  
        labels = kmeans[i].fit_predict(vectors)
        score_ss=sklearn.metrics.silhouette_score(vectors, labels, metric='euclidean')
        score_db=sklearn.metrics.davies_bouldin_score(vectors, labels)
        score_ch = sklearn.metrics.calinski_harabasz_score(vectors, labels)
        logging.warning("SS for {} = {}".format(nc[i],score_ss))
        logging.warning("DB for {} = {}".format(nc[i],score_db))
        logging.warning("CH for {} = {}".format(nc[i],score_ch))
Kmeans(kmeans,nc,vectors)
