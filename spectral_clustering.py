# ----------Spectral Clustering Algorithm------------
"""
1: Input: W ∈ R n×n, Number of clusters k & sigma.
2: Initialize: Compute the graph Laplacian L.
3: H ← matrix whose columns are the eigenvectors of L corresponding to the k-smallest eigenvalues.
4: r1, . . . , rn be the rows of H.
5: Cluster the points r1, . . . , rn using k-means algorithm.
6: Output: Clusters C1, . . . , Ck of the k-means algorithm.
"""

import pandas as pd
import numpy as np
import kmeans_clustering

def laplacianMatrix(df, sigma):
    N = df.shape[0]
    data = df.values
    L = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            t = 0
            for k in range(len(data[i])):
                t = t + (data[i][k] - data[j][k]) ** 2
            
            L[i][j] = - np.exp(- t / (sigma**2))
            L[j][i] = L[i][j]

    for i in range(N):
        L[i][i] = -sum(L[i])
        
    return L


def spectral_clustering(df, classes, sigma):
    L = laplacianMatrix(df, sigma)
    e, v = np.linalg.eig(L) # e: eigen values; v: corresponding eigen vectors
    sorted_index = np.argsort(e) 
    X = []
    for i in range(0,len(classes)):
        X.append(v[:, sorted_index[i]])
    
    Y = normaliseEachRowOfData(np.array(X).T)
    
    df_new = pd.DataFrame()
    key = 'DATA_{0}'
    for i in range(len(classes)):
        df_new[key.format(i)] = Y[:, i]
    Y_R, _ = kmeans_clustering.k_meams_clustering(df_new, classes)
    return Y_R

def laplacianMatrix1(df, sigma):
    N = df.shape[0]
    data = df.values
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            t = 0
            for k in range(len(data[i])):
                t = t + (data[i][k] - data[j][k]) ** 2
            
            A[i][j] = np.exp(- np.sqrt(t) / (sigma**2))
            A[j][i] = A[i][j]

    D = np.zeros((N, N))
    for i in range(N):
        D[i][i] = np.sqrt(sum(A[i]))
    
    D = np.linalg.inv(D)
    L = D.dot(A).dot(D)
    return L

def normaliseEachRowOfData(X):
    for i in range(len(X)):
        X[i] = X[i] / np.linalg.norm(X[i])
    return np.array(X)
