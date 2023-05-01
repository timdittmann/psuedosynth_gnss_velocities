#!/usr/bin/env python3

from itertools import product
from collections import deque

import os
import pandas as pd
import glob
import numpy as np

import datetime
from scipy import signal
import pywt

import pyarrow as pa
import pyarrow.parquet as pq
import json

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

import sys
from nested_xval_utils import *


fs={'feature':['psd_t'], 'stacking':['horizontal'], 'dims':[['H0','H1','UP']], 'augment':[True]}
feature_sets=[dict(zip(fs, v)) for v in product(*fs.values())]

#d = {'n_folds':[5],'max_depth': [50], 'n_estimators': [120], 'class_wt':[None],'wl_thresh':[0, 0.001,.005],}
#d = {'n_folds':[5],'max_depth': [10,100], 'n_estimators': [10,120], 'class_wt':[None, "balanced"],'wl_thresh':[-15,  0, 15]}
d = {'n_folds':[5],'max_depth': [5,50], 'n_estimators': [10,120], 'class_wt':[None, "balanced_subsample"],'wl_thresh':[-30, 0, 30]}
hyperp=[dict(zip(d, v)) for v in product(*d.values())]

print(hyperp)

import time
startTime = time.time()

###############
pq_list=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/',f) for f in os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/'))]
#pd_list=[pd.read_parquet(pq) for pq in pq_list if ".pq" in pq]
meta_list=[read_meta(pq_fs) for pq_fs in pq_list if ".pq" in pq_fs]
meta_df=pd.DataFrame.from_records(meta_list)


######################
ambient_list= list(meta_df[meta_df.magnitude.isnull()].eq_name.unique())
event_list=meta_df[~meta_df.magnitude.isnull()].sort_values(['magnitude'], ascending=False).groupby("eq_name").count().sort_values(['station'], ascending=False).index.tolist()
full_list=ambient_list+event_list

#convert to rsn
full_list=meta_df[meta_df.eq_name.isin(full_list)].record_number.unique()

for features in feature_sets:
    params=[i | features for i in hyperp]
    best_est_, stats=grid_search(full_list, params)

    X_train, y_train, name_list, times, snr_metric=list_to_featurearrays(full_list, best_est_, test=False) 
    print((X_train).shape)
    clf = RandomForestClassifier(n_estimators=best_est_['n_estimators'], max_depth=best_est_['max_depth'], class_weight=best_est_['class_wt'],random_state=42, n_jobs=-1).fit(X_train, y_train)
    
    keep_thresh=str(int(100*stats.threshold))
    
    joblib.dump(clf, '../models/synth_model_all_%s.pkl' %keep_thresh)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
    
    