#Load dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA
import seaborn
import os
import glob

def getPCAEigenPair(dataFrame):
    dataFrame.columns= ['Index','A','B','C','D']
    M=pd.pivot_table(dataFrame,index=['Index'])
    m = M.shape
    #print(M)
    df = M.replace(np.nan,0,regex=True)
    X_std = StandardScaler().fit_transform(df)
    print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))
    pca = PCA(n_components=4)
    print (pca)
    pca.fit_transform(df)
    print (pca.explained_variance_ratio_)
    component = pca.explained_variance_ratio_
    cov_mat = np.cov(X_std.T)
    eig_vals,eig_vecs=np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' %eig_vecs)
    print('\nEigenvalues \n%s' %eig_vals)
    #Explained variance
    #pca = PCA().fit(X_std)
    #egvco = np.empty(shape = [1,4])
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.xlabel('number of components')
    #plt.ylabel('cumulative explained variance')
    #plt.show()
    #egvco = np.concatenate([eig_vals])
    #return {"eigen_value" : eig_vals, "eigen_vector" : eig_vecs, "pcacomponents" : component, "egvc" : egvco}
    return {"eigen_value" : eig_vals, "eigen_vector" : eig_vecs}

file_key_list = ["p" + str(i+1) +".csv" for i in range(8)]
file_list = [d for d in file_key_list]
#eigen_val_array = np.array([])
componentN = 4
eigen_value_array = np.empty((componentN,0))
for key in file_list:
    print("key\n%s" %key)
    dfp=pd.read_csv(key)
    eigen_data = getPCAEigenPair(dfp)
    egval = np.array(eigen_data["eigen_value"])
    #s = np.array(eigen_data["pcacomponents"])
    #vt = np.array(eigen_data["eigen_vector"])
    #u = np.transpose(vt)
    #print(eigen_val_array)
    print(np.array(getPCAEigenPair(dfp)))
    #eigen_value_array=np.concatenate((eigen_value_array, np.array(getPCAEigenPair(dfp))))
    #A = u.dot(s.dot(vt))
    #print("")
    #print(A)
    #egvl = np.empty(shape = [1,4])
    #egvl = np.concatenate([eigen_data["egvc"]])
    #print (egvl)
    




# Visually confirm that the list is correctly sorted by decreasing eigenvalues
""" eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i])for i in range(len(eig_vals))]


#print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

pca = PCA(n_components=3)
print (pca)
pca.fit_transform(df1)
print (pca.explained_variance_ratio_)

#Explained variance
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show() """

