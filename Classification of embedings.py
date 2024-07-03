#!/usr/bin/env python
# coding: utf-8

# In[14]:


import torch
import numpy as np
import os


# In[15]:


device="mps" if torch.backends.mps.is_available() else "cpu"
device


# In[16]:


DATA_PATH="data"
TRAIN_EMBEDS = os.path.join(DATA_PATH, 'trainEmbeds.npz')
TEST_EMBEDS = os.path.join(DATA_PATH, 'testEmbeds.npz')
trainEmbeds, trainLabels = np.load(TRAIN_EMBEDS, allow_pickle=True).values()
testEmbeds, testLabels = np.load(TEST_EMBEDS, allow_pickle=True).values()
trainEmbeds.shape


# In[18]:


from sklearn.metrics import pairwise_distances
import pandas as pd
import seaborn as sns
sns.set()
def name_from_index(i, ClassList=ClassList):
    reversed_dict = {v: k for k, v in ClassList.items()}
    if isinstance(i, np.ndarray):  # Check if i is a NumPy array
        names = [reversed_dict.get(idx.item(), None) for idx in i]
        return names
    else:
        return reversed_dict.get(i, None)



def getDist(x, metric='euclidean', index=None, columns=None):
    dists = pairwise_distances(x, x, metric=metric)
    return pd.DataFrame(dists, index=index, columns=columns)

def heatmap(x, title='', cmap='Greens', linewidth=1):
    plt.figure(figsize=(17, 12))
    plt.title(title)
    sns.heatmap(x, cmap=cmap, square=True)
    plt.show()


# In[ ]:


# Note 88 first images are original and 4247 are augmented
# as long as to calculate (4335, 512) distance matrix is time consuming we get only distances of originals
inds = range(17595)

# Train embeddings
dists = getDist(trainEmbeds[inds], metric='euclidean', index=trainLabels[inds], columns=trainLabels[inds])
heatmap(dists, 'euclidean distance')

dists = getDist(trainEmbeds[inds], metric='cosine', index=trainLabels[inds], columns=trainLabels[inds])
heatmap(dists, 'cosine distance')


# In[ ]:


# Test embeddings
dists = getDist(testEmbeds, metric='euclidean', index=testLabels, columns=testLabels)
heatmap(dists, 'euclidean distance')

dists = getDist(testEmbeds, metric='cosine', index=testLabels, columns=testLabels)
heatmap(dists, 'cosine distance')


# In[ ]:


from sklearn.manifold import TSNE

inds = range(17595)
X_tsne1 = TSNE(n_components=2, init='pca', random_state=33).fit_transform(trainEmbeds[inds])
X_tsne2 = TSNE(n_components=2, init='random', random_state=33).fit_transform(trainEmbeds[inds])
y = [label for label in trainLabels[inds]]

fig, ax = plt.subplots(1, 2, figsize=(12, 8))

img = ax[0].scatter(X_tsne1[:, 0], X_tsne1[:, 1], c=y, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
ax[1].scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=y, alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))

ax[0].set_title('t-SNE with PCA initialization')
ax[1].set_title('t-SNE with random initialization')
plt.suptitle('Face embeddings')

cbar = plt.colorbar(img, ax=ax)
cbar.ax.set_yticklabels(np.unique(name_from_index(trainLabels[inds])))
plt.show()


# In[ ]:


# %%time
# X = np.copy(trainEmbeds)
# y = np.array([name_from_index(label) for label in trainLabels])
# from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# import warnings

# warnings.filterwarnings('ignore', 'Solver terminated early.*')

# param_grid = {'C': [1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'auto'],
#               'kernel': ['rbf', 'sigmoid', 'poly']}
# model_params = {'class_weight': 'balanced', 'max_iter': 10, 'probability': True, 'random_state': 3}
# model = SVC(**model_params)
# clf = GridSearchCV(model, param_grid)
# clf.fit(X, y)

# print('Best estimator: ', clf.best_estimator_)
# print('Best params: ', clf.best_params_)

