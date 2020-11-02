import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import logging
import numpy as np
import random
import sys
from sklearn.preprocessing import StandardScaler
import logging
scaler = StandardScaler()
vectors =np.load("Biobert_results_codes_models/Biobert_title_abstract.npy")
X1 = scaler.fit_transform(vectors)

def dbscan(vectors_docs,eps,min_pts,name):
  for(i in range(0,len(min_pts)):
    DBSCAN_model=sklearn.cluster.DBSCAN(eps=eps, min_samples=min_pts[i], metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30,n_jobs=-1)
    vectors_docs = scaler.fit_transform(vectors_docs)
    X=DBSCAN_model.fit_predict(vectors_docs)
    labels=DBSCAN_model.labels_.tolist()
    n_clusters= len(set(labels)) - (1 if -1 in labels else 0)
    logging.warning("Number of clusters: {}".format(n_clusters))
    score_ss=sklearn.metrics.silhouette_score(vectors_docs, X, metric='euclidean')
    score_db=sklearn.metrics.davies_bouldin_score(vectors_docs, X)
    score_ch = sklearn.metrics.calinski_harabasz_score(vectors_docs, X)
    logging.warning("SS for {} for eps {} = {}".format(name,eps,score_ss))
    logging.warning("DB for {} for eps {} = {}".format(name,eps,score_db))
    logging.warning("CH for {} for eps {} = {}".format(name,eps,score_ch))
  

dbscan(X1,20,[30,50,80,100],name="Biobert_title_Abstract")
dbscan(X1,30,[50,80,100],name="Biobert_title_Abstract")
dbscan(X1,40,[80,120,150,200],name="Biobert_title_Abstract")
