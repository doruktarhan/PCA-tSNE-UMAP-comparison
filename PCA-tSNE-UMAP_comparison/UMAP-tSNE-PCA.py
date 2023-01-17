import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from keras.datasets import fashion_mnist
import random




rgb_data = np.random.uniform(0,1,(1000,4)) 


perpexlity = [5,30,100]


plt.figure()
for i in range(3):

    perp = perpexlity[i]
    reduced_tsne = TSNE(n_components = 2,perplexity = perp).fit_transform(rgb_data)
    plt.subplot(1,3,i+1)
    plt.scatter(reduced_tsne[:,0],reduced_tsne[:,1],c = rgb_data)
    title = "perplexity = " + str(perp)  
    plt.title(title)    
plt.show()



plt.figure()
n_neighbors = [5, 20, 100]
min_dist = [0.05, 0.2, 0.8]


i = 0
j = 0 
for i in range(3):
    for j in range(3):
        plt.subplot(3,3, 3*i+j+1 )
        reduced_umap = UMAP(n_components=2, n_neighbors=n_neighbors[i], min_dist=min_dist[j]).fit_transform(rgb_data)
        plt.scatter(reduced_umap[:,0],reduced_umap[:,1],c = rgb_data)
        plt.title("n_neighbors = "+ str(n_neighbors[i]) + "  min_dist = " + str(min_dist[j]))
plt.show()



(trainX, trainy), (testX, testy) = fashion_mnist.load_data()


data_indexes = []

#loop for every instance 
for i in range(10):
    indexes = random.sample(range(6000*i , 6000*(i+1) ) ,600)
    data_indexes = data_indexes + indexes
 


train_x = trainX[data_indexes]/255
train_y = trainy[data_indexes]

train_x = train_x.reshape(6000,784)


#PCA
pca_reduce = PCA(n_components=2).fit_transform(train_x)
plt.figure()
plt.title('PCA Reduced Fashion MNIST Dataset')
for i in range(10):
    plt.scatter(pca_reduce[train_y==i,0], pca_reduce[train_y==i,1])
plt.show()


#t-SNE
plt.figure()
tsne_reduce = TSNE(n_components = 2,perplexity = 30).fit_transform(train_x)
for i in range(10):
    plt.scatter(tsne_reduce[train_y==i,0], tsne_reduce[train_y==i,1])
plt.title("t-SNE Reduced Fashion MNIST Dataset")    
plt.show()




#UMAP
plt.figure()
umap_reduced = UMAP(n_components=2, n_neighbors=20, min_dist=0.2).fit_transform(train_x)
for i in range(10):
    plt.scatter(umap_reduced[train_y==i,0], umap_reduced[train_y==i,1])
plt.title("UMAP Reduced Fashion MNIST Dataset")    
plt.show()










