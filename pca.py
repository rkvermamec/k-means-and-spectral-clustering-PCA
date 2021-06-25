# Principal Component Analysis algorithm for reducing the dimensionality of the points

import pandas as pd
import numpy as np

def pca(df, K = 2):
    normalisedVector = normalisedEigenVector(df)
    pc = reducedDimension(df, normalisedVector, K)
    return pc, normalisedVector

def normalisedEigenVector(df):
    X = df.values
    C = np.dot(X.T, X)
    e, v = np.linalg.eig(C) # e: eigen values; v: corresponding eigen vectors
    sorted_index = np.argsort(e)
    numberOfEV = len(e)
    normalisedVector = []
    # NORMALIZATION OF VECTOR
    for k in range(numberOfEV):
        norm = v[:, sorted_index[numberOfEV - 1 - k]]
        norm = norm / np.linalg.norm(norm)
        normalisedVector.append(norm)
    return normalisedVector

def reducedDimension(df, normalisedVector, K):
    X = df.values.T
    pc = []
    for k in range(K):
        nv = normalisedVector[k].T
        pct = []
        for i in range(df.shape[0]):
            pct.append(np.dot(nv, X[:, i]))
        pc.append(pct)
    pc = np.array(pc).T
    return pc

def reconstructionError(df, pc, normalisedVector, K):
    reconstructedData = []
    for k in range(K):
        if k == 0:
            for p in pc[:, k]:
                reconstructedData.append(np.dot(normalisedVector[k].T, p))
        else :
            i = 0
            for p in pc[:, k]:
                reconstructedData[i] += (np.dot(normalisedVector[k].T, p))
                i+=1
    
    reconstructedData = np.array(reconstructedData)
    originalData = df.values
    reconstructionError = 0
    for i in range(df.shape[0]):
        reconstructionError += np.sqrt(sum(np.square(reconstructedData[i] - originalData[i])))

    return reconstructionError

        

    