# PCA-tSNE-UMAP-comparison
Comparison of PCA, tSNE and UMAP feature reduction techniques 

There are two main approaches in reducing dimensionality. They are Projection and Manifold Learning. Projection methods tries to project every data point in high dimension to low dimension, while preserving the distance between them. In Manifold Learning, the algorithm works by modelling the manifold on which the training instances lie. The advantage of Manifold Learning methods that they are non-linear methods so that they can relate close samples to each other in a non-linear way with less effort. 

PCA is a projection algorithm which tries to identify the hyperplane which lies closest to the data and then projects the data onto that hyperplane while preserving the variance. The first components capture the most variance. 

t-SNE is a Manifold Learning method which tries to reduce a high dimensional data set to a low dimensional graph that retains a lot of the original information by giving each data point a location in two- or three-dimensional spaces. It is useful for visualization of high-dimensional datasets since it finds clusters in data. It tries to keep similar instances close and different instances apart. 

UMAP is also a Manifold Learning method which is very effective visualizing clusters or groups of data points and their relative proximities. The difference of UMAP to t-SNE is scalability which provides us to directly apply it onto sparse matrices. It is also faster than t-SNE. 
