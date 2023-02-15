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

from multiprocessing import Pool, cpu_count


def nested_xval_run(run, train_set_events,test_set_events):
    #############
    import time
    startTime = time.time()
    
    ######
    ###############
    '''
    pq_list=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/',f) for f in os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/'))]
    #pd_list=[pd.read_parquet(pq) for pq in pq_list if ".pq" in pq]
    meta_list=[read_meta(pq_fs) for pq_fs in pq_list if ".pq" in pq_fs]
    meta_df=pd.DataFrame.from_records(meta_list)
    '''
    meta_df=pd.read_csv('../data/meta/fs_meta.csv')
    
    ##########
    fs={'feature':['all','psd_t','psd', 'wavelet', 'time'], 'stacking':['horizontal'], 'dims':[['H0','H1','UP']], 'augment':[True, False]}
    fs={'feature':['psd_t','psd', 'time'], 'stacking':['horizontal'], 'dims':[['H0','H1','UP']], 'augment':[True, False]}
    #fs={'feature':['psd'], 'stacking':['horizontal'], 'dims':[['H0','H1','UP']], 'augment':[True]}
    feature_sets=[dict(zip(fs, v)) for v in product(*fs.values())]
    d = {'n_folds':[5],'max_depth': [10], 'n_estimators': [100], 'class_wt':[None,"balanced_subsample"], 'wl_thresh':[-30,0,30]} #'wl_thresh':[-15, -10, -5, 0]
    #d = {'n_folds':[5],'max_depth': [10], 'n_estimators': [10], 'class_wt':[None], 'wl_thresh':[15],} #'wl_thresh':[-15, -10, -5, 0]
    hyperp=[dict(zip(d, v)) for v in product(*d.values())]
    #########

    results=[]
    outer_results =[]
    first_tp=True
    first_fp=True
   

    #convert to rsn
    test_set=meta_df[meta_df.eq_name.isin(test_set_events)].record_number.unique()
    train_set=meta_df[meta_df.eq_name.isin(train_set_events)].record_number.unique()

    for features in feature_sets:
        params=[i | features for i in hyperp]
        best_est_, stats=grid_search(train_set, params)

        X_train, y_train, name_list, times, snr_metric=list_to_featurearrays(train_set, best_est_, test=False) 
        X_test, y_test, name_list, times, snr_metric=list_to_featurearrays(test_set, best_est_, test=True)
        clf = RandomForestClassifier(n_estimators=best_est_['n_estimators'], max_depth=best_est_['max_depth'], class_weight=best_est_['class_wt'],random_state=42, n_jobs=-1).fit(X_train, y_train)

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
        outer_results.append([p,r,f1,threshold, precisions, recalls, thresholds, y_test, y_pred_prob, best_est_, run, best_est_['feature'], test_set, features['augment']])
        df = pd.DataFrame(outer_results, columns=['precision','recall','f1','threshold','precisions','recalls','thresholds','y_act','y_prob','params', 'run', 'feature', 'test stations', 'augment'])
        df.to_csv('../data/results/nested_x_val_%s.csv' %run)
        
        # report progress
        print('RUN %s: %s : %s >f1=%.3f, %s, train shape:%s test shape%s' % (run, best_est_['feature'],best_est_['augment'],f1, stats, X_train.shape,X_test.shape)) 

        executionTime = (time.time() - startTime)
        print('RUN %s Execution time in hours: ' %run + str(executionTime/(60*60)))
        print(params)

        if features['augment']==True:
            
            ydf=pd.DataFrame([y_test,y_pred,y_pred_prob]).T
            ydf.columns = ydf.columns.astype(str)
            ydf.to_parquet('../data/results/aug/ydf_%s_%s.pq' %(features['feature'],run))
            
            xdf=pd.DataFrame(np.column_stack( (X_test,snr_metric)))
            xdf.columns = xdf.columns.astype(str)
            xdf.to_parquet('../data/results/aug/xdf_%s_%s.pq' %(features['feature'],run))
            
            if (features['feature']=='all') | (features['feature']=='psd_t'):
                joblib.dump(clf, '../models/aug/model_%s_run_%s.pkl' %(features['feature'],run))     
            
        '''
        if ((features['feature']=='all') & (features['augment']==False)):

            joblib.dump(clf, 'results/no_aug/model_run_%s.pkl' %run)
            
            ydf=pd.DataFrame([y_pred,y_test]).T
            ydf.columns = ydf.columns.astype(str)
            xdf=pd.DataFrame(np.column_stack( (X_test,snr_metric)))
            xdf.columns = xdf.columns.astype(str)
            
            ydf.to_parquet('results/aug/ydf_%s.pq' %run)
            xdf.to_parquet('results/aug/xdf_%s.pq' %run)
        '''
        #if (features['feature']=='all'):
        ##TEST INDIVIDUAL STATIONS
        meta_df_tmp=meta_df[meta_df.record_number.isin(test_set)]

        #Not noise stations
        p_df=meta_df_tmp.dropna(subset=['magnitude','Rrup'])

        #generate list of stations for event
        for index, row in p_df.iterrows():
            #sta=pd.concat([pd.read_parquet(pq) for pq in pq_list if "%s_%02d.pq" %(row.record_number,int(row.noise_lev)) in pq])
            pq_fn='../data/feature_sets/%s_%02d.pq' %(row.record_number,int(row.noise_lev))
            sta=pd.read_parquet(pq_fn)
            X_, y_, names, times, snr_metric=fs_to_Xy(sta, best_est_, test=True)

            y_pred_prob=clf.predict_proba(X_)[:, 1]
            y_pred = (y_pred_prob >= threshold).astype('int')

            snr_max=snr_metric[y_==1].max()

            arr=y_+y_pred

            if 2 in arr:
                marker='o'
                color='#377eb8'
                first_t_idx=np.argmax(arr>1)
                first_t=times.values[first_t_idx]
                
                if first_tp:
                    label='true positive'
                    first_tp=False
                else: 
                    label=None
            else:
                marker='x'
                color='#e41a1c'
                first_t=np.nan
                if first_fp:
                    label='false negative'
                    first_fp=False
                else:
                    label=None

            results.append([row.eq_name,row.record_number,row.magnitude, row.mechanism, row.Rrup, row.noise_lev, marker, \
                            color,label,threshold, snr_max, features['augment'], run, features['feature'], first_t])
            results_df2=pd.DataFrame(results, columns=['eq_name','record_number','magnitude','mechanism','Rrup','noise_lev','marker','color','label','threshold','snr_max','augment','run', 'features','first_t'])
            results_df2.to_csv('../data/results/station_results_%s.csv' %run)
    
    
def mp_handler():
    
    ###############
    pq_list=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/',f) for f in os.listdir(os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/'))]
    #pd_list=[pd.read_parquet(pq) for pq in pq_list if ".pq" in pq]
    meta_list=[read_meta(pq_fs) for pq_fs in pq_list if ".pq" in pq_fs]
    meta_df=pd.DataFrame.from_records(meta_list)
    meta_df.to_csv('../data/meta/fs_meta.csv', index=False)


    ######################
    ambient_list= list(meta_df[meta_df.magnitude.isnull()].eq_name.unique())
    event_list=meta_df[~meta_df.magnitude.isnull()].sort_values(['magnitude'], ascending=False).groupby("eq_name").count().sort_values(['station'], ascending=False).index.tolist()
    full_list=ambient_list+event_list

    ######################
    test_set_list=[]
    train_set_list=[]
    run_list=[]
    num_runs=10
    for k in np.arange(num_runs):
    #for k in np.arange(1):
        run=k+1
        items = deque(full_list)
        items.rotate(-k)
        test_set_events=list(items)[::num_runs]
        train_set_events=list(set(full_list) - set(test_set_events))
        test_set_list.append(test_set_events)
        train_set_list.append(train_set_events)
        run_list.append(run)
        
        nested_xval_run(run, train_set_events,test_set_events)
    
    #write function to check whats already written and not reprocess
    # read metadata of existing pq store to compile list of events-stations
    # difference two lists for residual to process
    
    #initiate pool to parallel process stations
    '''
    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(10)
    pool.starmap(nested_xval_run, zip(run_list,train_set_list,test_set_list))
    pool.close()
    '''

if __name__ == '__main__':
    mp_handler()
    
    
#nohup python -u ../bin/models/nest_xval_synth_MP.py > ../data/logs/program2.out 2>&1 &
