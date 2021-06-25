"""Implement the Principal Component Analysis algorithm for reducing the dimensionality of the points
given in the datasets: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.
data. Each point of this dataset is a 4-dimensional vector (d = 4) given in the first column of the datafile.
Reduce the dimensionality to 2 (k = 2). This dataset contains 3 clusters. Ground-truth cluster IDs are
given as the fifth column of the data file. In order to evaluate the performance of the PCA algorithm,
perform clustering (in 3 clusters) before and after dimensionality reduction using the Spectral Clustering
algorithm and then find the percentage of points for which the estimated cluster label is correct. Report
the accuracy of the Spectral Clustering algorithm before and after the dimensionality reduction. Report
the reconstruction error for k = 1, 2, 3."""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pca
import spectral_clustering

def plotData(df, classes):
    color = ['r', 'b', 'g']
    colorMap = {}
    i = 0
    for c in classes:
        colorMap[c] = color[i]
        colorMap[i] = color[i]
        i+=1
    
    plt.subplot(2, 2, 1)
    plt.scatter(df['POINT_0'], df['POINT_1'], c=df['Class'].map(lambda x: colorMap[x]), alpha=0.4)
    plt.title("Original Data")

    plt.subplot(2, 2, 2)
    plt.scatter(df['POINT_0'], df['POINT_1'], c=df['ClassifiedClassO'].map(lambda x: colorMap[x]), alpha=0.4)
    plt.title("Spectral On Original Data")

    plt.subplot(2, 2, 3)
    plt.scatter(df['POINT_0'], df['POINT_1'], c=df['ClassifiedClassR'].map(lambda x: colorMap[x]), alpha=0.4)
    plt.title("Spectral after dimensionality reduction")

    plt.show()  


def main_function(filePath):
    df = pd.read_csv(filePath[1], sep=",", names=['POINT_0', 'POINT_1', 'POINT_2', 'POINT_3', 'Class'])
    print(df.head())
    Y = df['Class']
    classes = df['Class'].unique().tolist()
    df = df[['POINT_0', 'POINT_1', 'POINT_2', 'POINT_3']]
    
    # SPECTRAL CLUSTERING ON ORIGINAL DATA
    SIGMA = 3
    Y_O = spectral_clustering.spectral_clustering(df, classes, SIGMA)
    
    # PCA ON ORIGINAL DATA FOR K=2
    K = 2
    pc, normalisedVector = pca.pca(df, K)
    pc = np.array(pc)
    rd_df = pd.DataFrame()
    for k in range(K):
        rd_df['POINT_{0}'.format(k)] = pc[:, k]

    print(rd_df.head())
    # SPECTRAL CLUSTERING ON dimensionality 2 DATA
    Y_R = spectral_clustering.spectral_clustering(rd_df, classes, SIGMA)

    # Reconstruction Error
    for i in range(1,5):
        p = pca.reducedDimension(df, normalisedVector, i)
        error = pca.reconstructionError(df, p, normalisedVector, i)
        print('Reconstruction Error For K={0} is {1:.12f}'.format(i, error))

    # DATA PLOTING AND ACCURACY
    df['Class'] = Y
    df['ClassifiedClassO'] = Y_O
    df['ClassifiedClassR'] = Y_R
    N = df.shape[0]
    print('\nClass wise performance on original data')
    print('------------------------------------------')
    result = performance(df, classes, 'Class', 'ClassifiedClassO')
    print('Over all performances: {0:.3f} %\n'.format(100*result/N))
    print('\nClass wise performance on 2-dimensional data')
    print('----------------------------------------------')
    result = performance(df, classes, 'Class', 'ClassifiedClassR')
    print('Over all performances: {0:.3f} %\n'.format(100*result/N))
    print('\nSpectral on original vs Reduced dimensionality data')
    print('------------------------------------------------------')
    result = performance(df, classes, 'ClassifiedClassO', 'ClassifiedClassR')
    print('Over all performances: {0:.3f} %\n'.format(100*result/N))
    
    plotData(df, classes)


def performance(df, classes, outputKey1, outputKey2):
    correctlyClassified = 0
    assignedColostur = []
    for c in classes:
        t = df[(df[outputKey1] == c)]
        totalRecord = t.shape[0]
        maxAssignedCluster = {'cls': -1, 'val': 0}
        for ct in classes:
            total = t[(t[outputKey2] == ct)].shape[0]
            if total > maxAssignedCluster['val'] and ct not in assignedColostur :
                maxAssignedCluster = {'cls': ct, 'val': total}
        
        correctlyClassified += maxAssignedCluster['val']
        assignedColostur.append(maxAssignedCluster['cls'])
        print('For class {0}, estimated cluster label correct percentage is {1:.2f} %'.format(c, 100*maxAssignedCluster['val']/totalRecord))

    return correctlyClassified


filePath = sys.argv
print (filePath)
main_function(filePath)