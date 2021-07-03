#Load dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA
from kmedoids import *
import seaborn
import os
import glob

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

                   
    
    
def getPCAEigenPair(dataFrame):
    components_n = 4
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
    #componentsdf.to_csv("100.csv")
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

#Load movie names and movie ratings
file_key_list = ["035_095" + str(i+1) +".csv" for i in range(8)]
file_list = [d for d in file_key_list]
componentN = 4
eigen_value_array = np.empty((componentN,0))
eigen_components = {}
count = 0
for key in file_list:
    new_key = count
    #print("key\n%s" %key)
    dfp=pd.read_csv(key)
    eigen_data = getPCAEigenPair(dfp)
    eigen_value_array = np.concatenate([eigen_value_array, eigen_data["eigen_value"]], axis=1)
    eigen_components[new_key] = eigen_data["eigen_vector"]
    count+=1

#print("keys file " + str(eigen_components.keys()) + "\n")
#for key in eigen_components:
    #print("key again file " + str(key) +"\n")
    #print(eigen_components[key])

weight_factor = eigen_value_array.mean(axis=1)
weight_factor = weight_factor/sum(weight_factor)
#print("weight")
#print(weight_factor)

distance_matrix, keys = formDistanceMatrix(eigen_components, weight_factor)
#print("distance matrix")
#print(distance_matrix)








