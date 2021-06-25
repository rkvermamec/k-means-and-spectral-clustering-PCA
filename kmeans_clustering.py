import pandas as pd
import numpy as np

def centroidsEvaluation(df, classes, classKey):
    centroids = {}
    for c in classes:
        tmpDf = df[df[classKey] == c]
        centroids[c] = {}
        for k in df.keys():
            if(k != classKey):
                centroids[c][k] = tmpDf[k].mean()

    return centroids

def classAssignmentByEuclideanDistance(df, classes, centroids):
    distance = {}
    for c in classes:
        for k in df.keys():
            if k in centroids[c].keys(): 
                distance[c] = (df[k] - centroids[c][k]) ** 2
        
        distance[c] = np.sqrt(distance[c])
    
    distanceDF = pd.DataFrame(distance)
    df['ClassifiedClass'] = distanceDF.idxmin(axis = 1)
    return df

def k_meams_clustering(df, classes):
    centroids = {}
    for c in classes:
        centroids[c] = {}
        r = np.random.randint(df.shape[0] - 1)
        for k in df.keys():
            centroids[c][k] = df[k][r]
    #centroids = centroidsEvaluation(dft, classes, 'Class')
    
    df = classAssignmentByEuclideanDistance(df, classes, centroids)
    while True:
        centroids = centroidsEvaluation(df, classes, 'ClassifiedClass')
        lastClassifiedClass = df['ClassifiedClass'].copy(deep=True)
        df = classAssignmentByEuclideanDistance(df, classes, centroids)
        if lastClassifiedClass.equals(df['ClassifiedClass']):
            break

    return df['ClassifiedClass'], centroids