import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn
import os
import glob
from variables import *
from kmedoids import *

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
    keys = components.keys()
    print("cluster keys " + str(keys) + "\n")
    initial_medoids = np.random.choice(list(range(0,len(keys))), num_clusters).tolist()
    km = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix', ccore=False)
    km.process()
    clusters = km.get_clusters()
    labels = [0]*len(keys)
    for i in range(len(clusters)):
        #print("i = " + str(i) + "\n")
        #print(clusters[i])
        for point in clusters[i]:
            labels[point] = i
    
    return keys, labels  


for key in eigen_components:
    #print("key\n%s" %key)
    #dfp=pd.read_csv(key)
    #eigen_data = getPCAEigenPair(dfp)
    #eigen_value_array = np.concatenate([eigen_value_array, eigen_data["eigen_value"]], axis=1)
    #print("*********")
    #print(eigen_value_array)
    #eigen_components[key] = eigen_data["eigen_vector"]
    a1 = np.array(eigen_data["eigen_vector"])
    b1 = np.array(eigen_data["eigen_vector"])
    w1 = weight_factor
    #print("Component Distance: ")
    #print(componentDistance(a1,b1,w1))

x = formDistanceMatrix(eigen_components,w1)
keys, labels = cluster(eigen_components, w1, distance_matrix, 3)
#print("keys: " + str(keys) + "\n")
#print("labels: " + str(labels) + "\n")








