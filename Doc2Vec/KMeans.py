import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import numpy as np
import random
import sys
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
nc=[5,10,15,20,40,60,80,100,120,150,180]
def Kmeans(kmeans,nc,X1,name):
    for i in range(len(kmeans)):
        X=kmeans[i].fit_predict(X1)
        sss1=sklearn.metrics.silhouette_score(X1,X)
        sdb1=sklearn.metrics.davies_bouldin_score(X1,X)
        score_ch = sklearn.metrics.calinski_harabasz_score(X1,X)
        logging.warning(name +" "+ "SS : "+ str(nc[i])+" "+ str(sss1))
        logging.warning(name +" "+ "DB : "+str(nc[i])+" " + str(sdb1))
        logging.warning(name +" "+ "CH : "+str(nc[i])+" " + str(score_ch))

model1_dm_300=np.load("/.../model1_dm_300.docvecs.vectors_docs.npy")
X1 = scaler.fit_transform(model1_dm_300)
kmeans = []
kmeans = [KMeans(n_clusters = i, n_init = 1000, max_iter = 3000) for i in nc]
Kmeans(kmeans,nc,X1,name="model1_dm_300")

model1_dbow_300=np.load("/.../model1_dbow_300.docvecs.vectors_docs.npy")
X2 = scaler.fit_transform(model1_dbow_300)
kmeans = []
kmeans = [KMeans(n_clusters = i, n_init = 2000, max_iter = 6000) for i in nc]
Kmeans(kmeans,nc,X2,name="model1_dbow_300")

model2_dm_300=np.load("/.../model2_dm_300.docvecs.vectors_docs.npy")
X3 = scaler.fit_transform(model2_dm_300)
kmeans = []
kmeans = [KMeans(n_clusters = i, n_init = 2000, max_iter = 6000) for i in nc]
Kmeans(kmeans,nc,X3,name="model2_dm_300")

model2_dbow_300=np.load("/.../model2_dbow_300.docvecs.vectors_docs.npy")
X4 = scaler.fit_transform(model2_dbow_300)
kmeans = []
kmeans = [KMeans(n_clusters = i, n_init = 2000, max_iter = 6000) for i in nc]
Kmeans(kmeans,nc,X4,name="model2_dbow_300")
