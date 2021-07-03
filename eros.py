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
from variables import *
from kmedoids import *
from silhouette import *
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import csv


distance = []
#count = 0
w1 = weight_factor
        
dm = formDistanceMatrix(eigen_components,w1)
maxlist = []
optimal_silhouette = {}
optimal_labels = {}
for x in range(2,9):
    silhouette = {}
    keylabels = {}
    for z in range(1,30):
        keys, labels = cluster(eigen_components, w1, distance_matrix,x)
        labels = np.array(labels)
        #labels.shape
        #print("x = " + str(x) + " z = " + str(z) + "\n")
        #print("keys: " + str(keys) + "\n")
        #print("labels: " + str(labels) + "\n")
        y = silhouette_score_block(distance_matrix, labels,metric = 'euclidean', sample_size = None, random_state=None, n_jobs=-1)
        #y = silhouette_score(distance_matrix,labels)
        dict_key = str(x) + ":" + str(z)
        #print("y = " + str(y))
        silhouette[dict_key] = y
        keylabels[dict_key] = labels
        
        
    maximum = max(silhouette, key=silhouette.get)
    #print("key = " + key)
    #print("maximum = " + maximum)
    #print(maximum, silhouette[maximum], keylabels[maximum])

    optimal_silhouette[maximum] = silhouette[maximum]
    optimal_labels[maximum] = keylabels[maximum]
    maxlist.append(optimal_silhouette[maximum])

optimal_maximum = max(optimal_silhouette, key=optimal_silhouette.get)
print(optimal_maximum, optimal_silhouette[optimal_maximum], optimal_labels[optimal_maximum])
print(maxlist)
maxdf = pd.DataFrame(maxlist)
maxdf.to_csv("df4035_095.csv")




















