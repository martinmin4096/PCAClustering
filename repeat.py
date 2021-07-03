import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import distance_metrics
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.externals.joblib import Parallel, delayed
from itertools import combinations
import seaborn
import os
import glob
from kmedoids import *
from silhouette import *
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import csv


def componentDistance(A, B, weightFactor):
    column_inner_product = (A*B).sum(axis=0)
    return np.sqrt(abs(2-2*np.sum(np.abs(column_inner_product)*weightFactor)))

def formDistanceMatrix(components, weightFactor):
    keys = components.keys()
    distance_matrix = np.empty((len(keys), len(keys)))
    for i in range(len(keys)):
        for j in range(len(keys)):
            distance_matrix[i][j] = componentDistance(components[keys[i]], components[keys[j]], weightFactor) if i!= j else 0
    return  distance_matrix, keys

def cluster(components, weightFactor, dmatrix, num_clusters):
    initial_medoids = np.random.choice(list(range(0,len(keys))), num_clusters).tolist()
    km = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix', ccore=False)
    km.process()
    clusters = km.get_clusters()
    labels = [0]*len(keys)
    for i in range(len(clusters)):
        for point in clusters[i]:
            labels[point] = i
    #print(labels)
    return keys, labels

def getPCAEigenPair(x, y, file, dataFrame):
    components_n = 3
    dataFrame.columns= ['Index','A','B','C','D']
    M=pd.pivot_table(dataFrame,index=['Index'])
    m = M.shape
    #print(M)
    df = M.replace(np.nan,0,regex=True)
    X_std = StandardScaler().fit_transform(df)
    #print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
    pca = PCA(n_components=None)
    #print (pca)
    pca.fit_transform(df)
    #print (pca.explained_variance_ratio_)
    componentslist = pca.explained_variance_ratio_
    componentsdf=pd.DataFrame(pca.explained_variance_ratio_)
    #csv_file = 
    componentsdf.to_csv("c" + file)
    #print(componentsdf)
    eig_vals = (pca.singular_values_[0:components_n]).reshape((components_n,1))
    eig_vecs = pca.components_.T[:, :components_n]
    #print('Eigenvectors \n%s' %eig_vecs)
    #print('\nEigenvalues \n%s' %eig_vals)
    #Explained variance
    #pca = PCA().fit(X_std)
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance')
    #plt.show()
    return {"eigen_value" : eig_vals, "eigen_vector" : eig_vecs}

def PCArepeat(filename):
    r = 0
    i = 0
    componentN = 3
    eigen_value_array = np.empty((componentN,0))
    eigen_components = {}
    count = 0
    file = filename
    for r in range(6):
        file_key_list = [file + str(r+1) + "_" + str(i+1) + ".csv" for i in range(8)]
        file_list = [d for d in file_key_list]
        for key in file_list:
            #print str(key)
            new_key = count
            #print("key\n%s" %key)
            dfp=pd.read_csv(key)
            eigen_data = getPCAEigenPair(r+1, i+1, key, dfp)
            eigen_value_array = np.concatenate([eigen_value_array, eigen_data["eigen_value"]], axis=1)
            eigen_components[new_key] = eigen_data["eigen_vector"]
        weight_factor = eigen_value_array.mean(axis=1)
        weight_factor = weight_factor/sum(weight_factor)
        distance_matrix, keys = formDistanceMatrix(eigen_components, weight_factor)
    









