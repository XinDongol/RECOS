#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:19:25 2016

@author: simon
"""# This is the implement of Fig 4
"""Anchor vectors in LeNet-5 for the MNIST dataset change during training:"""
###########
# Note: this part is drawing paraments from LeNet-5.


import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt

# set up subplots
fig, ax = plt.subplots(nrows=1,ncols=2, sharey=True)

# process and draw change of anchor vectors during training.
def draw_conv(dx,pl,name,y=1):
    size= dx[0].shape
    dxanchor = np.zeros(size[0]*size[1]*size[2])

    # vectorize all tensor of filter 
    for i in range(len(dx)):
        dxanchor = np.vstack((dxanchor, dx[i].reshape((size[0]*size[1]*size[2],size[3])).T[0]))
    print(dxanchor.shape)

    # normalization them and make them l2-norm = 1
    dxanchor = preprocessing.normalize(dxanchor, norm='l2')
    
    # change: cos(\theta): inner produce of normalized vectors
    change = []
    for j in range(dxanchor.shape[0]):
        if j >0:
            change += [np.inner(dxanchor[j],dxanchor[j-1])]

    #plot change
    pl.plot(range(len(change)), np.array(change), linewidth=3, alpha=0.5) 
    #plot y=1
    pl.plot(range(len(change)), np.ones(len(change)), linewidth=1.5, color = 'g') 

    # set x-axis and y-axis, title
    pl.set_ylim(0.999999, 1.0000001)
    #pl.set_xlim(1, 300)
    pl.set_title(name)
    if y==1:
        pl.set_ylabel('Change of Anchor Vector')
    pl.set_xlabel('Iterations')
    pl.yaxis.get_major_formatter().set_powerlimits((0,1)) 
    return change

# never used, never sure about whether this part of codes work
def draw_fc(dx,pl,name):
    size= dx[0].shape
    dxanchor = np.zeros(size[1])
    for i in range(len(dx)):
        dxanchor = np.vstack((dxanchor, dx[i].reshape((size[0]*size[1]*size[2],size[3])).T[0]))
    print(dxanchor.shape)
    dxanchor = preprocessing.normalize(dxanchor, norm='l2')
    
    change = []
    for j in range(dxanchor.shape[0]):
        if j >0:
            change += [np.inner(dxanchor[j],dxanchor[j-1])]
        
    pl.plot(range(len(change)), np.array(change)) 
    pl.set_ylim(0.999999, 1)
    #pl.xlim(1, 300)
    pl.set_title(name)
    

    
change1 = draw_conv(w1_list, ax[0], 'Conv-1')
change2 = draw_conv(w2_list, ax[1], 'Conv-2',0)
#draw_fc(w3_list,ax[2],'Fc-3')
