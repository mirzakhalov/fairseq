import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
enc_df = pd.read_csv("enc_tok.csv", header=0,names = ['label', 'features'])

features = enc_df['features'].values
labels = enc_df['label'].values
labels = np.array(labels)
X = []

for feature in features:
    temp = feature[1:-1].split()
    X.append([float(temp[0]), float(temp[1])])

#print(X)


wcss=[]
#this loop will fit the k-means algorithm to our data and 
#second we will compute the within cluster sum of squares and #appended to our wcss list.
for i in range(1,11): 
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
# plt.plot(range(1,11),wcss)
# plt.title('The Elbow Method Graph')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


kmeans = KMeans(n_clusters=5, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
# We are going to use the fit predict method that returns for each #observation which cluster it belongs to. The cluster to which #client belongs and it will return this cluster numbers into a #single vector that is  called y K-means
y_kmeans = kmeans.fit_predict(X)

X = np.array(X)

indices1 = (y_kmeans==0).nonzero()[0]
indices2 = (y_kmeans==1).nonzero()[0]
indices3 = (y_kmeans==2).nonzero()[0]
indices4 = (y_kmeans==3).nonzero()[0]
indices5 = (y_kmeans==4).nonzero()[0]


plt.scatter(X[indices1, 0], X[indices1, 1], s=10, c='red', label ='Cluster 1')
plt.text(X[indices1, 0] + 0.3, X[indices1, 1] + 0.3, labels[indices1], fontsize = 9 )
plt.scatter(X[indices2, 0], X[indices2, 1], s=10, c='blue', label ='Cluster 2')
plt.scatter(X[indices3, 0], X[indices3, 1], s=10, c='green', label ='Cluster 3')
plt.scatter(X[indices4, 0], X[indices4, 1], s=10, c='cyan', label ='Cluster 4')
plt.scatter(X[indices5, 0], X[indices5, 1], s=10, c='magenta', label ='Cluster 5')
#Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')
plt.show()

