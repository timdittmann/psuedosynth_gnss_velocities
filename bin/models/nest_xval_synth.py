#!/usr/bin/env python3

#MODIFIED to store all stations, with column for feature type

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


fs={'feature':['psd', 'wavelet', 'time', 'all'], 'stacking':['horizontal'], 'dims':[['H0','H1','UP']], 'augment':[True, False]}
fs={'feature':['psd'], 'stacking':['horizontal'], 'dims':[['H0','H1','UP']], 'augment':[True]}
feature_sets=[dict(zip(fs, v)) for v in product(*fs.values())]

#d = {'n_folds':[5],'max_depth': [50], 'n_estimators': [120], 'class_wt':[None],'wl_thresh':[0, 0.001,.005],}
d = {'n_folds':[5],'max_depth': [10,100], 'n_estimators': [10,100], 'class_wt':[None,"balanced"], 'wl_thresh':[-15,0,15],} #'wl_thresh':[-15, -10, -5, 0]
d = {'n_folds':[5],'max_depth': [10], 'n_estimators': [10], 'class_wt':[None], 'wl_thresh':[15],} #'wl_thresh':[-15, -10, -5, 0]
hyperp=[dict(zip(d, v)) for v in product(*d.values())]

print(hyperp)

###############
pq_list=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/',f) for f in os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/'))]
#pd_list=[pd.read_parquet(pq) for pq in pq_list if ".pq" in pq]
meta_list=[read_meta(pq_fs) for pq_fs in pq_list if ".pq" in pq_fs]
meta_df=pd.DataFrame.from_records(meta_list)


######################
ambient_list= list(meta_df[meta_df.magnitude.isnull()].eq_name.unique())
event_list=meta_df[~meta_df.magnitude.isnull()].sort_values(['magnitude'], ascending=False).groupby("eq_name").count().sort_values(['station'], ascending=False).index.tolist()
full_list=ambient_list+event_list

print(len(full_list))
#############
import time
startTime = time.time()

aug_y_pred_keep=[]
aug_y_test_keep=[]
aug_x_test_keep=[]
aug_snr_test_keep=[]

noaug_y_pred_keep=[]
noaug_y_test_keep=[]
noaug_x_test_keep=[]
noaug_snr_test_keep=[]

results=[]
outer_results =[]

first_tp=True
first_fp=True
    
#Nested Cross validation, 10 runs
num_runs=10
num_runs=2
#for k in np.arange(num_runs):
for k in np.arange(num_runs):
    run=k+1
    items = deque(full_list)
    items.rotate(-k)
    test_set_events=list(items)[::num_runs]
    train_set_events=list(set(full_list) - set(test_set_events))
    
    #convert to rsn
    test_set=meta_df[meta_df.eq_name.isin(test_set_events)].record_number.unique()
    train_set=meta_df[meta_df.eq_name.isin(train_set_events)].record_number.unique()
    
    for features in feature_sets:
        params=[i | features for i in hyperp]
        print(params)
        best_est_, stats=grid_search(train_set, params)
        
        X_train, y_train, name_list, times, snr_metric=list_to_featurearrays(train_set, best_est_, test=False) 
        X_test, y_test, name_list, times, snr_metric=list_to_featurearrays(test_set, best_est_, test=True)
        clf = RandomForestClassifier(n_estimators=best_est_['n_estimators'], max_depth=best_est_['max_depth'], class_weight=best_est_['class_wt'],random_state=10, n_jobs=-1).fit(X_train, y_train)

        #y_pred=clf.predict(X_test)
        y_pred_prob=clf.predict_proba(X_test)[:, 1]

        #threshold=0.5 
        threshold=stats.threshold # Hyper Param from xval training
        y_pred = (y_pred_prob >= threshold).astype('int')
        
        
        ###
        # evaluate the model on test data
        p, r, f1, blah=precision_recall_fscore_support(y_test, y_pred, average='binary')
        #
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

        # store the result
        outer_results.append([p,r,f1,threshold, precisions, recalls, thresholds, y_test, y_pred_prob, best_est_, run, best_est_['feature'], test_set, features['augment'],features['feature']])
        #'precision','recall','f1','threshold','precisions','recalls','thresholds','y_act','y_prob','params', 'run', 'feature', 'test stations', 'augment', 'features'
        # report progress
        print('>f1=%.3f, %s' % (f1, stats)) 
        
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
        
        #if ((features['feature']=='all') & (features['augment']==True)):
        if ((features['feature']=='psd') & (features['augment']==True)):
    
            joblib.dump(clf, '../models/aug/model_run_%s.pkl' %run)
            
            aug_y_pred_keep.append(y_pred)
            aug_y_test_keep.append(y_test)
            aug_x_test_keep.append(X_test)
            aug_snr_test_keep.append(snr_metric)
        #if ((features['feature']=='all') & (features['augment']==False)):
        if ((features['feature']=='psd') & (features['augment']==False)):
    
            joblib.dump(clf, '../models/no_aug/model_run_%s.pkl' %run)
            
            noaug_y_pred_keep.append(y_pred)
            noaug_y_test_keep.append(y_test)
            noaug_x_test_keep.append(X_test)
            noaug_snr_test_keep.append(snr_metric)
        
        #if (features['feature']=='all'):
        ##TEST INDIVIDUAL STATIONS
        meta_df_tmp=meta_df[meta_df.record_number.isin(test_set)]

        #Not noise stations
        p_df=meta_df_tmp.dropna(subset=['magnitude','Rrup'])

        #generate list of stations for event
        for index, row in p_df.iterrows():
            sta=pd.concat([pd.read_parquet(pq) for pq in pq_list if "%s_%02d.pq" %(row.record_number,int(row.noise_lev)) in pq])

            X_, y_, names, times, snr_metric=fs_to_Xy(sta, best_est_, test=True)

            y_pred_prob=clf.predict_proba(X_)[:, 1]
            y_pred = (y_pred_prob >= threshold).astype('int')

            snr_max=snr_metric[y_==1].max()

            arr=y_+y_pred

            if 2 in arr:
                marker='o'
                color='#377eb8'
                #count+=1
                if first_tp:
                    label='true positive'
                    first_tp=False
                else: 
                    label=None
            else:
                marker='x'
                color='#e41a1c'

                if first_fp:
                    label='false negative'
                    first_fp=False
                else:
                    label=None

            results.append([row.eq_name,row.record_number,row.magnitude, row.mechanism, row.Rrup, row.noise_lev, marker, \
                            color,label,threshold, snr_max, features['augment'], run, features['feature']])

        
    df = pd.DataFrame(outer_results, columns=['precision','recall','f1','threshold','precisions','recalls','thresholds','y_act','y_prob','params', 'run', 'feature', 'test stations', 'augment', 'features'])
    df.to_csv('../data/results/nested_x_val.csv')
    
    results_df2=pd.DataFrame(results)
    results_df2.to_csv('../data/results/station_results.csv')

y_pred_data=(np.concatenate( aug_y_pred_keep, axis=0 ))
y_test_data=(np.concatenate( aug_y_test_keep, axis=0 ))
x_test_data=np.concatenate( aug_x_test_keep, axis=0 )
snr_test_data=np.concatenate( aug_snr_test_keep, axis=0 )

ydf=pd.DataFrame([y_pred_data,y_test_data]).T
#xdf=pd.DataFrame(x_test_data)
xdf=pd.DataFrame(np.column_stack( (x_test_data,snr_test_data)))

ydf.to_parquet('../data/results/aug/ydf.pq')
xdf.to_parquet('../data/results/aug/xdf.pq')



y_pred_data=(np.concatenate( noaug_y_pred_keep, axis=0 ))
y_test_data=(np.concatenate( noaug_y_test_keep, axis=0 ))
x_test_data=np.concatenate( noaug_x_test_keep, axis=0 )
snr_test_data=np.concatenate( noaug_snr_test_keep, axis=0 )

ydf=pd.DataFrame([y_pred_data,y_test_data]).T
#xdf=pd.DataFrame(x_test_data)
xdf=pd.DataFrame(np.column_stack( (x_test_data,snr_test_data)))

ydf.to_parquet('../data/results/no_aug_/ydf.pq')
xdf.to_parquet('../data/results/no_aug_/xdf.pq')



    
#nohup python -u nest_xval_synth.py > program.out5 2>&1 &