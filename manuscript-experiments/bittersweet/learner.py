import os
import dill
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from collections import defaultdict

def specificity(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    speci = 1 - fpr
    return speci[1]

class Learner():
    def __init__(self, md, gs, taste, name, config_params):
        self.features = list(md.columns)[6:]
        self.md, self.gs = md, gs
        self.X, self.y = md.iloc[:, 6:], md['taste']
        self.X_gs, self.y_gs = gs.iloc[:, 7:], gs['taste']
        self.taste = taste
        self.name = name
        self.config_params = config_params
        
    def plot_gs_results(self, cv_results, config_params):
        cv_df = pd.DataFrame(cv_results)
        params = defaultdict(set)
        for p in cv_results['params']: 
            for param, value in p.items():
                params[param].add(value)
                
        grid_params = ['param_' + p for p,v in params.items() if len(v) > 1]
        
        if len(grid_params) == 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 20))
            sns.scatterplot(y='mean_test_' + config_params['refit'],
                            x=grid_params[0],
                            data=cv_df,
                            ax=ax1
                           )
            sns.scatterplot(y='mean_train_' + config_params['refit'],
                            x=grid_params[0] ,
                            data=cv_df,
                            ax=ax2
                           )
            
        elif len(grid_params) == 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 20))
            sns.scatterplot(y='mean_test_' + config_params['refit'],
                            x=grid_params[0],
                            hue=grid_params[1],
                            data=cv_df,
                            ax=ax1
                           )
            sns.scatterplot(y='mean_train_' + config_params['refit'],
                            x=grid_params[0],
                            hue=grid_params[1],
                            data=cv_df,
                            ax=ax2
                           )
            
        elif len(grid_params) == 3:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 20))
            sns.scatterplot(y='mean_test_' + config_params['refit'],
                            x=grid_params[0],
                            hue=grid_params[1],
                            style=grid_params[2],
                            data=cv_df,
                            ax=ax1,
                           )
            sns.scatterplot(y='mean_train_' + config_params['refit'],
                            x=grid_params[0],
                            hue=grid_params[1],
                            style=grid_params[2],
                            data=cv_df,
                            ax=ax2
                           )
        else:
            pass
            
    def grid_search(self, clf, hyperparameters, plot=False):
        config_params = self.config_params
        model = GridSearchCV(clf, hyperparameters, **config_params)
        model.fit(self.X, self.y)
        
        self.grid_obj = model
        self.grid_results = model.cv_results_
        self.model = model.best_estimator_

        # Plot grid search results
        if plot:
            self.plot_gs_results(model.cv_results_, config_params)        
        
    def evaluate_gs(self, threshold=0.5, threshold_search=True):
        config_params = self.config_params
        cv_df = pd.DataFrame(self.grid_results)
        model = self.model
        X_gs, y_gs = self.X_gs, self.y_gs
        gs, md = self.gs, self.md
        
        # Log for generating tables later on
        log = {
            'mean_cv_' + s: cv_df['mean_test_' + s].iloc[0] for s in config_params['scoring']
        }

        #
        y_score = model.predict_proba(X_gs)
        
        if threshold_search:
            for t in np.arange(0.25, 0.52, 0.01):
                y_pred = y_score[:, 1] > t
                print(t)
                print(classification_report(y_gs, y_pred))                
        else:
            if self.taste == 'Sweet':
                thresh_vals = list()
                for t in np.arange(0.3, 0.7, 0.01):
                    y_pred = y_score[:, 1] > t
                    thresh_vals.append([((metrics.recall_score(y_gs, y_pred) +\
                                          metrics.recall_score(~y_gs,~y_pred)) / 2.0), t])


                mx_val = max(thresh_vals, key=lambda k: k[0])[0]
                print(mx_val)
                shortlisted_thresh = [t[1] for t in thresh_vals if t[0] == mx_val]

                threshold=min(shortlisted_thresh, key= lambda k: abs(0.5 - k))
                print("Threshold: ", threshold)
               
            self.threshold=threshold
            y_pred = y_score[:, 1] > threshold
            
        log['test' + '_sensitivity'] = metrics.recall_score(y_gs, y_pred)
        log['test' + '_specificity'] = specificity(y_gs, y_pred)
        log['test' + '_f1'] = metrics.f1_score(y_gs, y_pred)
        log['test' + '_roc_auc'] = metrics.roc_auc_score(y_gs, y_score[:, 1])
        log['test' + '_average_precision'] = metrics.average_precision_score(y_gs, y_score[:, 1])

        
        print(classification_report(y_gs, y_pred))
#         x = metrics.confusion_matrix(y_gs, y_pred)
#         log['test_tn'], log['test_fp'], log['test_fn'], log['test_tp'] = x[0][0], x[0][1], x[1][0], x[1][1] 

        
        if self.taste == 'Bitter':
            for ref in set(gs['reference']):
                print(ref)
                subset = (gs.reference == ref) & (gs['In Bitter Domain'])
                print(classification_report(y_gs[subset], y_pred[subset]))
#                 x = metrics.confusion_matrix(y_gs[subset], y_pred[subset])
#                 log[ref + '_tn'], log[ref + '_fp'], log[ref + '_fn'], log[ref + '_tp'] = x[0][0], x[0][1], x[1][0], x[1][1] 
                log[ref + '_sensitivity'] = metrics.recall_score(y_gs[subset], y_pred[subset])
                log[ref + '_specificity'] = specificity(y_gs[subset], y_pred[subset])
                log[ref + '_f1'] = metrics.f1_score(y_gs[subset], y_pred[subset])
                log[ref + '_roc_auc'] = np.nan if ref == 'Bitter-New' else metrics.roc_auc_score(y_gs[subset], y_score[subset, 1])
                log[ref + '_average_precision'] = metrics.average_precision_score(y_gs[subset], y_score[subset, 1])
                
        return log
        
    def savemodel(self, loc):
        loc = os.path.join(loc, self.taste.lower())
        if not os.path.exists(loc):
            os.mkdir(loc)
        dill.dump(self, open(os.path.join(loc, self.name), 'wb'))
        
        
        