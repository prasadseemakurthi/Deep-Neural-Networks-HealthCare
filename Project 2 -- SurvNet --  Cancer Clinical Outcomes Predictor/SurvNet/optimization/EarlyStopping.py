# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 22:35:05 2016

@author: Safoora Yousefi
"""
import numpy as np
def isOverfitting(results, interval=5, num_intervals = 3):
    flag = True
    end = len(results)
    maxIter = len(results)-1
    for i in range(num_intervals - 1):
        begin1 = end - (i + 1) * interval
        end1 = end - (i) * interval
        begin2 = end - (i + 2) * interval
        if np.mean(results[begin1:end1]) > np.mean(results[begin2:begin1]):
            flag = False
    if flag:
        maxIter = np.argmax(results[begin2:end1]) + end - interval * num_intervals
    return flag, maxIter
