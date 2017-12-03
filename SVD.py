# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 11:16:55 2017

@author: Q
"""

import numpy as np
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def eulidSim(inA,inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))
def pearsSim(inA,inB):
    if len(inA) < 3:
        return 1
    return 0.5 + 0.5 * np.corrcoef(inA,inB,rowvar = 0)[0][1]
def cosSim(inA,inB):
    num = float(inA.T*inB)
    demon = np.linalg.norm(inA)*np.linalg.norm(inB)
    return 0.5 + 0.5 * (num/demon)
def svdEst(dataSet,user,simMeans,item):
    n = np.shape(dataSet)[1]
    U,sigma,VT = np.linalg.svd(dataSet)
    sig4 = np.mat(np.eye(4)*sigma[:4])
    xformedItems = dataSet.T * U[:,:4] * sig4
#    xformedItems = VT[:4,:].T
    simTotal = 0
    ratSimTotal = 0
    for i in range(n):
        userRating = dataSet[user,i]
        if userRating == 0 or i == item:
            continue
        similarity = simMeans(xformedItems[i,:].T,xformedItems[item,:].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
    
    
    
def recommend(dataSet,user,N=11,simMeans = cosSim,estMethod = svdEst):
    unratedItems = np.nonzero(dataSet[user,:]==0)[1]
    if len(unratedItems) == 0:
        return 'every dish have been rate'
    itemScores = [] 
    for item in unratedItems:
        estimatedScore = estMethod(dataSet,user,simMeans,item)
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key = lambda pp: pp[1],reverse = True)[:N]
           
           
           
data = np.mat(loadExData2())
#u,sigma,vt = np.linalg.svd(data)
#sig4 = np.mat(np.eye(4)*sigma[:4])
#
#xform = data.T*u[:,:4]*sig4
#xform2 = sig4 * vt[:4,:]    
estimatedScore = recommend(data,6)




