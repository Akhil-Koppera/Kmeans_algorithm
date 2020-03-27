import pandas as pd
import numpy as np



def Kmeans(X_train,N):
    np.random.seed(42)
  
    centers =X_train[np.random.choice(X_train.shape[0], N, replace=False), :]
    updatedCenters= deepcopy(centers)

    initialCenters = np.zeros(centers.shape) 
    clusters = np.zeros(X_train.shape[0])
    distances = np.zeros((X_train.shape[0],N))

    dist_centers = np.linalg.norm(updatedCenters - initialCenters)

    while dist_centers != 0:
        i = 0;
        while i<N:
            distances[:,i] = np.linalg.norm(X_train - centers[i], axis=1)
            i = i+1;
        initialCenters = deepcopy(updatedCenters)
        clusters = np.argmin(distances, axis = 1)
        j =0;
        while j<N:
            updatedCenters[j] = sum(X_train[clusters == j])/len(X_train[clusters == j])
            j = j+1;
        dist_centers = np.linalg.norm(updatedCenters - initialCenters) 
    df1= pd.DataFrame(X)
    df1['clusters']=clusters
    Clusters=[]
    for i in range(K):
        idx = df1['clusters'] == i
        xi=df1[idx] 
        xi=xi.drop(columns = xi.columns[len(xi.columns)-1]).to_numpy()
        Clusters.append(xi)
    return Clusters
