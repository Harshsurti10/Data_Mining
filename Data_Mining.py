# importing necessary libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest


"""Dataset Loading"""

dataset_loc = 'X_train.txt'

dataset = []
with open(dataset_loc) as f:
  for line in f:
    words = line.split()
    float_words = []
    for word in words:
      float_words.append(float(word))
    dataset.append(float_words)
dataset

labels_path = 'y_train.txt'
labels = []

with open(labels_path) as f:
  for line in f:
    labels.append(float(line))

print(labels)

def get_label_indices(label):
  indices = []
  for index, val in enumerate(labels):
    if val == label:
      indices.append(index)
  return indices

def get_features(label):
  features = []
  indices = get_label_indices(label)
  for index in indices:
    features.append(dataset[index])
  return features

""" 1. KMeans Clustering"""

def get_kmeans_clusters(activity):
  feature_maps = np.array(get_features(activity))
  model = KMeans(n_clusters = 2)
  results = model.fit_predict(feature_maps)
  clusters = np.unique(results)
  return clusters, results, feature_maps

def show_clusters(clusters, results, feature_maps, title):
  graph_label = "normal"
  for cluster in clusters:
    if cluster == 0 or cluster == -1:
      graph_label = "Outlier"
    else:
      graph_label = "Normal"
    index = np.where(results == cluster)
    plt.scatter(feature_maps[index, 0], feature_maps[index, 1], label=graph_label)
  plt.legend()
  plt.title(title)
  plt.show()

clusters, results, feature_maps = get_kmeans_clusters(5)
show_clusters(clusters, results, feature_maps,"KMeans Clusters")

"""2. Isolation Forest Algorithm"""

from sklearn.ensemble import IsolationForest

def get_iForest_clusters(label):
  feature_maps = np.array(get_features(label))
  model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.1), 
                        max_features=1.0, verbose=0, random_state=50)
  model.fit(feature_maps)
  results = model.predict(feature_maps)
  clusters = np.unique(results)
  return clusters, results, feature_maps
 
cluster, results, feature_maps = get_iForest_clusters(5)
show_clusters(clusters, results, feature_maps,"IForest Clusters")