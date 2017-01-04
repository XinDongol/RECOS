#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:39:10 2016

@author: X Dong
"""
## This is the implement of Fig 8: How Anchor vectors in AlexNet changes when going deeper

import numpy as np
from sklearn.manifold import TSNE
from sklearn import preprocessing
from matplotlib import pyplot as plt
import os
from pylab import *
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf


# load parameters of AlexNet
net_data = load("bvlc_alexnet.npy").item()

# add sub plots 
fig, ax = plt.subplots(nrows=2,ncols=2)
colors = np.random.rand(700)

w1 = net_data["conv1"][0]
print(w1.shape)

#reshape tensor of filter (4-D to 1-D)
dx = np.reshape(w1,(w1.shape[0]*w1.shape[1]*w1.shape[2],w1.shape[3]))

#normalize anchor vector
dx = preprocessing.normalize(dx, norm='l2')
print(dx.shape)

# TSNE function from Sklearn, there is also a modified version by me in my github.
model = TSNE(n_components=2, random_state=0)  # set n_components as 2 for we want to visualize them in 2-D space
np.set_printoptions(suppress=True)
a=model.fit_transform(dx) # a is the processed anchor vector. 1-D (x1,x2)
print(a.shape)
ax[0][0].axis([-60, 60, -60, 60])

# here, we set alpha(transparency) as 0.3 for transparency can reflect the degree of clustering
# The deeper color is , the better clustering is.
ax[0][0].scatter(a[:, 0], a[:, 1],c=np.random.rand(363), alpha=0.3, s=50)
ax[0][0].set_title('Conv-1')  


# staring process anchor vectors in next layer
w3 = net_data["conv3"][0]
print(w3.shape)
dx = np.reshape(w3,(w3.shape[0]*w3.shape[1]*w3.shape[2],w3.shape[3]))
dx = preprocessing.normalize(dx, norm='l2')
print(dx.shape)
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
a=model.fit_transform(dx) 
ax[0][1].axis([-20, 20, -20, 20])
ax[0][1].scatter(a[:, 0], a[:, 1],c=np.random.rand(2304), alpha=0.3, s=50)
ax[0][1].set_title('Conv-3')  



w5 = net_data["conv5"][0]
print(w5.shape)
dx = np.reshape(w5,(w5.shape[0]*w5.shape[1]*w5.shape[2],w5.shape[3]))
dx = preprocessing.normalize(dx, norm='l2')
print(dx.shape)
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
a=model.fit_transform(dx) 
ax[1][0].axis([-30, 30, -30, 30])
ax[1][0].scatter(a[:, 0], a[:, 1],c=np.random.rand(1728), alpha=0.3, s=50)
ax[1][0].set_title('Conv-5')  


# compared term
# we randomly generate anchor vectors whose elements are i.i.d. normal distribution
w6 = np.random.normal(loc=0.0, scale=1.0, size=(500,500))
dx = preprocessing.normalize(w6, norm='l2')
print(dx.shape)
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
a=model.fit_transform(dx) 
ax[1][1].axis([-30, 30, -30, 30])
ax[1][1].scatter(a[:, 0], a[:, 1],c=np.random.rand(500), alpha=0.3, s=50)
ax[1][1].set_title('Gaussian')  



plt.show() 