import sklearn
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import logging
import numpy as np
import random
import sys
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
def dbscan(vectors_docs,eps,name):
  min_pts=10
  DBSCAN_model=sklearn.cluster.DBSCAN(eps=eps, min_samples=min_pts, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30)
  vectors_docs = scaler.fit_transform(vectors_docs)
  X=DBSCAN_model.fit_predict(vectors_docs)
  labels=DBSCAN_model.labels_.tolist()
  n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  logging.warning("Number of clusters: {}".format(n_clusters))
  score_ss=sklearn.metrics.silhouette_score(vectors_docs, X, metric='euclidean')
  score_db=sklearn.metrics.davies_bouldin_score(vectors_docs, X)
  score_ch = sklearn.metrics.calinski_harabasz_score(vectors_docs, X)
  logging.warning("SS for {} for eps {} = {}".format(name,eps,score_ss))
  logging.warning("DB for {} for eps {} = {}".format(name,eps,score_db))
  logging.warning("CH for {} for eps {} = {}".format(name,eps,score_ch))

#Notable Results only found with model1_dm with eps 5 and 10
model1_dm_300=np.load("Doc2vec_results_code_models/doc2vec_models/model1_dm_300.docvecs.vectors_docs.npy")
X1 = scaler.fit_transform(model1_dm_300)
dbscan(X1,10,name="model1_dm_300")
"""
model1_dbow_300=np.load("Doc2vec_results_code_models/doc2vec_models/model1_dbow_300.docvecs.vectors_docs.npy")
X2 = scaler.fit_transform(model1_dbow_300)
dbscan(X2,10,name="model1_dbow_300")

model2_dm_300=np.load("Doc2vec_results_code_models/doc2vec_models/doc2vec_models/model2_dm_300.docvecs.vectors_docs.npy")
X3 = scaler.fit_transform(model2_dm_300)
dbscan(X3,24,name="model2_dm_300")

model2_dbow_300=np.load("Doc2vec_results_code_models/doc2vec_models/model2_dbow_300.docvecs.vectors_docs.npy")
X4 = scaler.fit_transform(model2_dbow_300.docvecs.vectors_docs)
dbscan(X4,23,name="model2_dbow_300")
"""
