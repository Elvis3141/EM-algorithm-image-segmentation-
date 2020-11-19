# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:16:24 2020

@author: Elvis Shehu 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

#read image
image_file = 'im.jpg'
original_image = cv2.imread(image_file)
# Converting from BGR Colours Space to HSV
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
#convert 3d image to 2d
img_2d = img.reshape((-1,3))
# convert to np.float32
X = np.float32(img_2d)

# EM algorithm 
def EM(X, k, max_iter, tol):    
    n, m = X.shape    
    gamma = np.full(shape=X.shape, fill_value=1/k)    
    row = np.random.randint(low=0, high=n, size=k)
    mean = [X[r_i,:] for r_i in row]
    cov_matrix = [np.cov(X.T) for _ in range(k)]        
    mixing_coeff = np.full(shape=k, fill_value=1/k)
    mixture_component = np.zeros((n,k))
    log_likehood = []
    log_lk_prev = 0    
    for iter_ in range(max_iter):
        # E-Step
        p_x = None
        for i in range(k):
            mixture_component[:,i] = multivariate_normal(mean=mean[i], cov=cov_matrix[i]).pdf(X)
            phi_N =  mixing_coeff * mixture_component 
            p_x = phi_N.sum(axis=1)[:, np.newaxis]            
            gamma = phi_N / p_x         
        # M-Step
        for i in range(k):
            gamma_i = gamma[:, [i]]
            total_gamma_i = gamma_i.sum()
            mean[i] = (X * gamma_i).sum(axis=0) / total_gamma_i
            cov_matrix[i] = np.cov(X.T, aweights=(gamma_i/total_gamma_i).flatten(), bias=True)
            mixing_coeff[i] = gamma_i.sum() / len(gamma_i)          
        #log-likelihood
        log_lk_curr = np.sum(np.log(p_x))
        log_likehood.append(log_lk_curr)
        #check convergence
        if np.abs(log_lk_curr - log_lk_prev) < tol:
            break
        log_lk_prev = log_lk_curr            
    return gamma, mean, log_likehood

#Image error
def error(X_true, X_pred):
    N = X_true.shape[0]    
    error = np.sum((X_true-X_pred)**2) / N
    return error
 
#plot log-likelihood
def plot_cost(costs):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Log-Likelihood")
    plt.show() 
#plot image
def plot_image(img):
    figure_size = 10
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,2),plt.imshow(img_3d)
    plt.title('K = %i' % k), plt.xticks([]), plt.yticks([])
    plt.show()     


# number of clusters
k = 8
max_iter = 5
tol = 1e-2
# X=image, K=clusters, max_iter = max itaterations, tol = tolerance
gamma, means, log_likehood= EM(X, k, max_iter, tol)
#get prediction
labels = np.argmax(gamma, axis=1)
#convert 2d to 3d
centers = np.uint8(means)
X_pred = centers[labels]
img_3d = X_pred.reshape((img.shape))
#calculate error
error = error(img_2d, X_pred)
#plot log-likelihood     
plot_cost(error)
#plot image
plot_image(img_3d)
print("Image Errror :",error, "\nIterations :", len(log_likehood))



   




























