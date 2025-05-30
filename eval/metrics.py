import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import pairwise_distances 
from scipy.linalg import sqrtm,cholesky,eigh
from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import mode 

def fidelity(Kx,Ky):

    Dx, Vx = eigh(Kx)

    Dx = np.clip(Dx,a_min=0,a_max=None)
    
    sqrtx = (Vx * np.sqrt(Dx)) @ Vx.T

    Dxy,Vxy = eigh(sqrtx @ Ky @ sqrtx)
    Dxy = np.clip(Dxy,a_min=0,a_max=None)
    sqrtxy = (Vxy * np.sqrt(Dxy)) @ Vxy.T

    return np.trace(sqrtxy)

def neural_bures_sim(X,Y):

    n1,d1 = X.shape
    n2,d2 = Y.shape

    c1 = np.eye(n1) - np.ones((n1,n1))/n1 
    c2 = np.eye(n2) - np.ones((n2,n2))/n2

    
    Kx,Ky = c1 @ X @ X.T @ c1, c2 @ Y @ Y.T @ c2 

    fid = fidelity(Kx,Ky)
    #print(fid)

    return fid/np.sqrt(np.trace(Kx)*np.trace(Ky))

def global_linear_error(X,Y):

      model = LR().fit(X,Y)
      yhat = model.predict(X)

      err = np.linalg.norm(yhat - Y,axis=1)

      return err 

def find_nearest_neighbors(X,Y,n_neighbors=20):
     
    dists_x = pairwise_distances(X,n_jobs=2)
    dists_y = pairwise_distances(Y,n_jobs=2)

    nbrs_x = NN(n_neighbors=n_neighbors + 1,algorithm='auto',metric='precomputed').fit(dists_x)
    nbrs_y = NN(n_neighbors=n_neighbors + 1,algorithm='auto',metric='precomputed').fit(dists_y)

    _,top_neighbors_x = nbrs_x.kneighbors(dists_x)
    _,top_neighbors_y = nbrs_y.kneighbors(dists_y)
    
    top_neighbors_x,top_neighbors_y = top_neighbors_x[:,1:],top_neighbors_y[:,1:]

    total_overlap = np.zeros((X.shape[0],))
    
    for jj in range(n_neighbors):

         total_overlap += (top_neighbors_x[:,jj:jj+1] == top_neighbors_y).sum(axis=1)

    perc_overlap = total_overlap/n_neighbors

    return perc_overlap
         

def assess_gmm_fit(y,yhat):
     

    true_labels = np.unique(y)

    precisions,recalls=[],[]
    for label in true_labels:
         
        inds = y == label
        pred_labels = yhat[inds]
        most_common_label,_ = mode(pred_labels,nan_policy='omit')
        print(f"label {label} corresponds to gmm cluster {most_common_label}")
        pred_inds = yhat == most_common_label
        
        true_positives = np.sum(inds*pred_inds) # indices should be 2 if both are 1, 1 if one is, 0 if neither is
        false_positives = np.sum((1 - inds) * pred_inds)

        true_negatives = np.sum((1-inds) * (1-pred_inds))
        false_negatives = np.sum(inds *(1 -pred_inds)) 

        prec = true_positives/(true_positives + false_positives)
        rec = true_positives/(true_positives + false_negatives)
        
        precisions.append(prec)
        recalls.append(rec)

    return precisions,recalls

def get_all_stats(true_latents,pred_latents,true_labels,pred_labels):


    nbs = neural_bures_sim(true_latents,pred_latents)
    global_lin_errs = global_linear_error(true_latents,pred_latents)
    nn_neighors = find_nearest_neighbors(true_latents,pred_latents)

    precision,recall = assess_gmm_fit(true_labels,pred_labels)

    return nbs,global_lin_errs,nn_neighors,precision,recall


def get_all_stats(true_latents,pred_latents,true_labels,pred_labels):


    nbs = neural_bures_sim(true_latents,pred_latents)
    global_lin_errs = global_linear_error(true_latents,pred_latents)
    nn_neighors = find_nearest_neighbors(true_latents,pred_latents)

    precision,recall = assess_gmm_fit(true_labels,pred_labels)

    return nbs,global_lin_errs,nn_neighors,precision,recall

