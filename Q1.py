# Q 1

"""
Implement the k-means and spectral clustering algorithms for clustering the points given in the datasets:
http://cs.joensuu.fi/sipu/datasets/jain.txt. Plot the obtained results. In order to evaluate the
performance of these algorithms, find the percentage of points for which the estimated cluster label is
correct. Report the accuracy of both the algorithm. The ground truth clustering is given as the third
column of the given text file.

"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import kmeans_clustering
import spectral_clustering

def plotData(df, centroids):
    color = ['r', 'b', 'g']
    colorMap = {}
    i = 0
    plt.subplot(2, 2, 1)
    for c in centroids.keys():
        plt.scatter(centroids[c]['X'],centroids[c]['Y'], color=color[i], s=80, edgecolor='black')
        colorMap[c] = color[i]
        i += 1
    plt.scatter(df['X'], df['Y'], c=df['ClassifiedClassK'].map(lambda x: colorMap[x]), alpha=0.2)
    plt.title("K-means clustering")

    plt.subplot(2, 2, 2)
    plt.scatter(df['X'], df['Y'], c=df['ClassifiedClassS'].map(lambda x: colorMap[x]), alpha=0.2)
    plt.title("Spectral clustering")

    plt.subplot(2, 2, 3)
    totalRecord = df.shape[0]
    correctlyClassified = df[(df['Class'] == df['ClassifiedClassK'])].shape[0]
    wrong = totalRecord - correctlyClassified
    if wrong > correctlyClassified:
        wrong = correctlyClassified
        correctlyClassified = totalRecord - wrong
    piData = [[correctlyClassified, wrong],
      ['correct ({0})'.format(correctlyClassified), 'Incorrect ({0})'.format(wrong)]]
    plt.pie(piData[0], labels = piData[1])
    print('Estimated cluster label correct percentage (K MEANS)= {0:.2f} %'.format((correctlyClassified / totalRecord) * 100))
    
    plt.subplot(2, 2, 4)
    totalRecord = df.shape[0]
    correctlyClassified = df[(df['Class'] == df['ClassifiedClassS'])].shape[0]
    wrong = totalRecord - correctlyClassified
    if wrong > correctlyClassified:
        wrong = correctlyClassified
        correctlyClassified = totalRecord - wrong
    piData = [[correctlyClassified, wrong],
      ['correct ({0})'.format(correctlyClassified), 'Incorrect ({0})'.format(wrong)]]
    plt.pie(piData[0], labels = piData[1])
    print('Estimated cluster label correct percentage (Spectral)= {0:.2f} %'.format((correctlyClassified / totalRecord) * 100))
    plt.show()  


def performance(df, classes, outputKey):
    for c in classes:
        t = df[(df['Class'] == c)]
        totalRecord = t.shape[0]
        maxAssignedCluster = {'cls': -1, 'val': 0}
        for ct in classes:
            total = t[(t[outputKey] == ct)].shape[0]
            if total > maxAssignedCluster['val']:
                maxAssignedCluster = {'cls': ct, 'val': total}
        print('For class {0}, estimated cluster label correct percentage is {1:.2f} %'.format(c, 100*maxAssignedCluster['val']/totalRecord))

    
def main_function(filePath):
    df = pd.read_csv(filePath[1], sep="\t", names=['X', 'Y', 'Class'])
    Y = df['Class']
    classes = df['Class'].unique()
    df = df[['X', 'Y']]
    
    # K-MEANS CLUSTERING
    Y_K, centroids = kmeans_clustering.k_meams_clustering(df, classes)
    #print('final centroids K MEANS: ',centroids)
    
    # SPECTRAL CLUSTERING
    SIGMA = 1
    Y_S = spectral_clustering.spectral_clustering(df[['X', 'Y']], classes, SIGMA)
    
    # DATA PLOTING AND ACCURACY
    df['Class'] = Y
    df['ClassifiedClassK'] = Y_K
    df['ClassifiedClassS'] = Y_S
    print('\nClass wise performance K Means')
    print('------------------------------')
    performance(df, classes, 'ClassifiedClassK')
    print('\nClass wise performance SPECTRAL')
    print('-------------------------------')
    performance(df, classes, 'ClassifiedClassS')
    print('\n')
    plotData(df, centroids)

filePath = sys.argv
print (filePath)
main_function(filePath)