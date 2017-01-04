#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:51:15 2016

@author: X Dong
"""

# This is the implement of Fig 2
"""Left: Original data-points without batch normalization are crowded in the first quadrant
and have various length varying in a wide range. Right: Data-points after batch normalization
have basically the same length and are relatively evenly distributed on a circle."""



import numpy as np
import scipy.io as sio  
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

# simplified batch normalization: no scale and bias parameter
# Actually, scale and bias do not have too much inportance for this conclusion.
def normal(data):
    #data is np.array
    shape=data.shape
    data_mean=np.sum(data, axis=0)/(shape[0]+0.00001)
    print '\n data_mean:'
    #print data_mean
    a=data-np.outer(np.ones(shape[0]).T,data_mean)
    var=np.sum(a*a, axis=0)/(shape[0]+0.00001)
    #print var
    c=a/np.outer(np.ones(shape[0]).T,np.sqrt(var))
    print np.sum(c, axis=0)
    print '\n var:'
    print np.sum(c*c, axis=0)/(shape[0]+0.00001)
    return c

# plot 3D scatter, hard to recognize, abandoned!
def plot_data_3d(data,llo=0,blo=6):
    plt.figure()
    ax=plt.subplot(111,projection='3d')
    trans=data.T
    ax.scatter(trans[0],trans[1],trans[2],s=80,c=['m','c','r','b'])
    ax.set_zlabel('Z') 
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_xlim3d(llo,blo)
    ax.set_ylim3d(llo,blo)
    ax.set_zlim3d(llo,blo)
    plt.show()
    

# plot 2D scatter
colors = np.random.rand(20)
def plot_data_2d(pltd, data, name,llo=0,blo=6):
    trans=data.T
    area = [70,90,110,130,190]
#    colors = np.random.rand(20)

# use different colors and size, and make it easy to recognize
    pltd.scatter(trans[0],trans[1],c=colors, alpha=0.5, s=area) 
    pltd.plot([0,0],[-100,100],'b-')
    pltd.plot([-100,100],[0,0],'b-')
    pltd.axis([llo,blo,llo,blo])
    pltd.set_title(name)

    
#generate two dimensions data
# This data can also be generated from the input of MNIST
# use 'mnist.train.images', please import mnist first!!
def gener_data(num,d,rang):
    data=np.zeros(d)
    for i in range(num):
        rarray=rang*np.random.random(size=d)
        data=np.vstack((data,rarray))
    print '\n data:'
    print data[1:num+1]
    return data[1:num+1]

fig, ax = plt.subplots(nrows=1,ncols=2)

#data=gener_data(20,2,5)
plot_data_2d(ax[0], data, 'Original Data-points', llo=-2, blo=6)
#data_nor=normal(data)
plot_data_2d(ax[1], data_nor, 'Data-points after BN', llo=-2,blo=2)

# generate data for circle
def circle(x,y,r):
    xarr=[]
    yarr=[]
    for i in range(1000):
        jiao=float(i)/1000*2*math.pi
        x1=x+r*math.cos(jiao)
        y1=y+r*math.sin(jiao)
        xarr.append(x1)
        yarr.append(y1)
    return xarr,yarr


cx, cy = circle(0,0,1.5)
ax[1].plot(cx,cy,linestyle='--', color='r', alpha=0.5)