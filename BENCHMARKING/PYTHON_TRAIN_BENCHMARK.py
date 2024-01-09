import math
import os, sys
import numpy as np
from scipy import ndimage
import netCDF4 as nc4
import re
import h5py

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def load_h5(loc):
    dataset = h5py.File(loc, 'r')
    X = np.array(dataset['X'][:])
    Y = np.array(dataset['Y'][:])

    print('X: ', X.shape)
    print('Y: ', Y.shape)
    return(X,Y)

def train(x, y):
    rfmodel = RandomForestClassifier(n_estimators = 21, max_depth = 14, n_jobs = -1,random_state = 50, class_weight = 'balanced')
    rfmodel.fit(x.T, y)
    
X_train,Y_train = load_h5("./data/raw_Vortex.h5")

timeit -n 2 train(X_train, Y_train) 