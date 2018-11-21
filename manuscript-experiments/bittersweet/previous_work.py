import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import classification_report

alpha = 1.5

class N3:
    def __init__(self, X, y, alpha):
        
        # Range Scale
        preproc = MinMaxScaler()
        preproc.fit(X)
        self.X_trn = preproc.transform(X)
        self.y_trn = y
        self.preproc = preproc
        
        # Save other params
        self.alpha = alpha
        
    def _score(self, X_tst, training_set):            
        X_trn, y_trn = self.X_trn, self.y_trn
        X_tst = self.preproc.transform(X_tst)
        
        # Find similarities by ranks
        simbyrank = np.zeros((X_tst.shape[0], X_trn.shape[0]))
        
        for i in range(X_tst.shape[0]):
            dist = ((X_trn - X_tst[i, :]) ** 2).mean(1)
            
            if training_set:
                dist[i] = 10000000
                    
            
            ranks = rankdata(dist) ** self.alpha
            simbyrank[i, :] = (1 / (1+dist)) / ranks

        # Sweet weights 
        is_sweet = y_trn == 'Sweet'
        sweet_weights = simbyrank[:, is_sweet].sum(1) / (is_sweet).sum()
        sweet_weights = sweet_weights / np.sum(sweet_weights)
        
        # Non-sweet weights
        nsweet_weights = simbyrank[:, ~is_sweet].sum(1) / (~is_sweet).sum()
        nsweet_weights = nsweet_weights / np.sum(nsweet_weights)

        return sweet_weights, nsweet_weights
    
    def predict_proba(self, X_tst):
        sweet_weights, nsweet_weights = self._score(X_tst)
        
        return np.exp(-sweet_weights) / (np.exp(-sweet_weights) + np.exp(-nsweet_weights))
                                
    def predict(self, X_tst, training_set=False):
        sweet_weights, nsweet_weights = self._score(X_tst, training_set)
        predictions = np.array(['Non-sweet']*X_tst.shape[0])
        p = sweet_weights > nsweet_weights
        predictions[p] = 'Sweet'
        
        return predictions
    
    
    
    