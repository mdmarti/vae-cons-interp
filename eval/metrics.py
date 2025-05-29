import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrix import pairwise_distances 
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import mode 

def fidelity(Kx,Ky):

    sqrtx = sqrtm(Kx)
    sqrtxy = sqrtm(sqrtx @ Ky @ sqrtx)

    return np.trace(sqrtxy)

def neural_bures_sim(X,Y):

    n1,d1 = X.shape
    n2,d2 = Y.shape

    c1 = np.eye(n1) - np.ones((n1,n1))/n1 
    c2 = np.eye(n2) - np.ones((n2,n2))/n2

    Kx,Ky = c1 @ X @ X.T @ c1, c2 @ Y @ Y.T @ c2 

    fid = fidelity(Kx,Ky)

    return fid/np.sqrt(np.trace(Kx)*np.trace(Ky))

def global_linear_error(X,Y):

      yhat = LR().fit_predict(X,Y)

      err = np.linalg.norm(yhat - Y,axis=1)

      return err 

def find_nearest_neighbors(X,Y,n_neighbors=20):
     
    dists_x = pairwise_distances(X,n_jobs=2)
    dists_y = pairwise_distances(Y,n_jobs=2)

    nbrs_x = NN(n_neighbors=n_neighbors + 1,algorithm='ball_tree',metric='precomputed').fit(dists_x)
    nbrs_y = NN(n_neighbors=n_neighbors + 1,algorithm='ball_tree',metric='precomputed').fit(dists_y)

    top_neighbors_x,_ = nbrs_x.neighbors(dists_x)
    top_neighobrs_y,_ = nbrs_y.neighbors(dists_y)
    top_neighbors_x,top_neighobrs_y = top_neighbors_x[:,1:],top_neighobrs_y[:,1:]

    total_overlap = np.zeros((X.shape[0],))
    for jj in range(n_neighbors):
         
         total_overlap += (top_neighbors_x[:,jj:jj+1] == top_neighobrs_y).sum(axis=1)

    perc_overlap = total_overlap/n_neighbors

    return perc_overlap
         

def assess_gmm_fit(y,yhat):
     

    true_labels = np.unique(y)

    precisions,recalls=[],[]
    for label in true_labels:
         
        inds = true_labels == label
        pred_labels = yhat[inds]
        most_common_label,_ = mode(pred_labels,nan_policy='omit')
        pred_inds = yhat == most_common_label

        true_positives = np.sum((inds == 1) and (pred_inds == 1))
        false_positives = np.sum((inds == 0) and (pred_inds == 1))

        true_negatives = np.sum((inds == 0) and (pred_inds == 0))
        false_negatives = np.sum((inds == 1) and (pred_inds == 0))

        prec = true_positives/(true_positives + false_positives)
        rec = true_positives/(true_positives + false_negatives)

        precisions.append(prec)
        recalls.append(rec)

    return precisions,recalls









