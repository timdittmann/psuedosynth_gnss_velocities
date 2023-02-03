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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

import sys




def read_meta(path):
    custom_meta_key = 'feature_meta'
    table = pq.read_table(path)
    meta_json = table.schema.metadata[custom_meta_key.encode()]
    meta = json.loads(meta_json)
    return meta


def grid_search(train_set,params):
    '''
    '''
    result=[]
    for param in params: 
        precision, recall, f1, threshold = k_fold_results(train_set, param)
        result.append((list(param.values()))+list([precision,recall,f1,threshold]))
    df = pd.DataFrame(result, columns=list(param.keys())+['precision','recall','f1','threshold'])
    print(df.sort_values(by=['f1'], ascending=False)[['max_depth', 'n_estimators','class_wt','threshold','wl_thresh','f1']]) #
    i=df.f1.idxmax()
    return(params[i],df.iloc[i])

def grid_search_jgr(train_set, params, X_train_nga, y_train_nga, jgr_test, include_nga):
    '''
    '''
    result=[]
    for param in params: 
        precision, recall, f1, threshold = k_fold_results_jgr(train_set, param,X_train_nga, y_train_nga,jgr_test, include_nga)
        result.append((list(param.values()))+list([precision,recall,f1,threshold]))
    df = pd.DataFrame(result, columns=list(param.keys())+['precision','recall','f1','threshold'])
    print(df.sort_values(by=['f1'], ascending=False)[['max_depth', 'n_estimators','class_wt','threshold','wl_thresh','f1']]) #
    i=df.f1.idxmax()
    return(params[i],df.iloc[i])

def k_fold_results(train_set, param):
    '''
    cross validate using k-fold approach #https://scikit-learn.org/stable/modules/cross_validation.html
    can't use scikit learn api because I want to keep the event/station pairs seperated
    so wrote own to return precision, recall and f1
    '''
    split_val=int((1/param['n_folds'])*len(train_set))

    stats=[]
    thresh_list=[]

    for i in np.arange(param['n_folds']):
        #test_fold=train_set[i*split_val:i*split_val+split_val]
        # Modify to take in 'ordered list' (ambient set then ranked events) and take every 5th one for test/train split
        items = deque(train_set)
        items.rotate(-i)
        test_fold=list(items)[::5]
        train_fold=list(set(train_set) - set(test_fold))
        
        X_train, y_train, name_list, times, snr_metric=list_to_featurearrays(train_fold, param, test=False)  
        X_test, y_test, name_list, times, snr_metric=list_to_featurearrays(test_fold, param, test=True)
        #print((X_train.shape,X_test.shape))
        
        # train classifier
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], class_weight=param['class_wt'],random_state=10, n_jobs=-1).fit(X_train, y_train)
        
        ###WAS JUST THIS 
        #y_pred=clf.predict(X_test)
        
        y_pred_prob=clf.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        threshold=thresholds[ix]

        y_pred_prob=clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype('int')
        
        precision, recall, f1, blah=precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        stats.append((precision, recall, f1, threshold))
        #thresh_list.append(threshold)
    precision=np.array(stats)[:,0].mean()
    recall=np.array(stats)[:,1].mean()
    f1=np.array(stats)[:,2].mean()
    threshold=np.array(stats)[:,3].mean()
    
    #thresh=np.array(thresh_list).mean()
    
    return precision, recall, f1, threshold

def k_fold_results_jgr(train_set, param, X_train_nga, y_train_nga, jgr_test, include_nga):
    '''
    cross validate using k-fold approach #https://scikit-learn.org/stable/modules/cross_validation.html
    can't use scikit learn api because I want to keep the event/station pairs seperated
    so wrote own to return precision, recall and f1
    '''
    split_val=int((1/param['n_folds'])*len(train_set))

    stats=[]
    thresh_list=[]

    for i in np.arange(param['n_folds']):
        #test_fold=train_set[i*split_val:i*split_val+split_val]
        # Modify to take in 'ordered list' (ambient set then ranked events) and take every 5th one for test/train split
        items = deque(train_set)
        items.rotate(-i)
        test_fold=list(items)[::5]
        train_fold=list(set(train_set) - set(test_fold))
        
        test_set_df=jgr_test[jgr_test.eventID.isin(test_fold)]
        train_set_df=jgr_test[jgr_test.eventID.isin(train_fold)]
        
        X_train, y_train, name_list, times, snr_metric=list_to_featurearrays_JGR(train_set_df,param, test=True) 
        
        if include_nga:
            X_train=np.vstack((X_train,X_train_nga))
            y_train=np.hstack((y_train,y_train_nga))
        
        X_test, y_test, name_list, times, snr_metric=list_to_featurearrays_JGR(test_set_df,param, test=True)
        #print((X_train.shape,X_test.shape))
        
        # train classifier
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], class_weight=param['class_wt'],random_state=10, n_jobs=-1).fit(X_train, y_train)
        
        ###WAS JUST THIS 
        #y_pred=clf.predict(X_test)
        
        y_pred_prob=clf.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        threshold=thresholds[ix]

        y_pred_prob=clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= threshold).astype('int')
        
        precision, recall, f1, blah=precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        stats.append((precision, recall, f1, threshold))
        #thresh_list.append(threshold)
    precision=np.array(stats)[:,0].mean()
    recall=np.array(stats)[:,1].mean()
    f1=np.array(stats)[:,2].mean()
    threshold=np.array(stats)[:,3].mean()
    
    #thresh=np.array(thresh_list).mean()
    
    return precision, recall, f1, threshold

def list_to_featurearrays(record_number_list, param, test):
    '''
    convert list of station/events to concatenated dataframes
    then convert dataframes to feature sets using feature extraction function
    '''
    #fold=pd.read_parquet([os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/',f) for f in fold])
    
    fold_set_list=[]
    
    # if no augmentation, on training runs only take the lowest noise slices plus the ambient noise
    if ((param['augment'] == False) & (test==False)): 
        path=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/', '%s_*05.pq' %f) for f in record_number_list]
        for j in path:
            fold_set_list+=glob.glob(j)
        path=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/', '%s_*a.pq' %f) for f in record_number_list]
        for j in path:
            fold_set_list+=glob.glob(j)
        
    else: 
        path=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets/', '%s_*' %f) for f in record_number_list]
        for j in path:
            fold_set_list+=glob.glob(j)
    
    fold=pd.read_parquet(fold_set_list)
    #X_, y_, name_list, times=fs_to_Xy_horizontal(fold, params)
   
    
    X_, y_, name_list, times, snr_metric=fs_to_Xy(fold, param, test)


    return X_, y_, name_list, times, snr_metric


def list_to_featurearrays_JGR(df,param, test=True):
    '''
    convert list of station/events to concatenated dataframes
    then convert dataframes to feature sets using feature extraction function
    '''    
    fold_set_list=[]
    
    fold_set_list=[]
    for i,row in df.iterrows():
        fn=os.path.join(os.path.dirname(os.getcwd()),'data/feature_sets_JGR',row.eventID+'_'+row.station+'.pq')
        if os.path.exists(fn):
            fold_set_list.append(fn)

    fold=pd.read_parquet(fold_set_list)
    #X_, y_, name_list, times=fs_to_Xy_horizontal(fold, params)
   
    X_, y_, name_list, times, snr_metric=fs_to_Xy(fold, param, test)


    return X_, y_, name_list, times, snr_metric

def list_to_featurearrays_ambient_test(param, test):
    '''
    convert list of station/events to concatenated dataframes
    then convert dataframes to feature sets using feature extraction function
    '''
    #fold=pd.read_parquet([os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/',f) for f in fold])
    
    fold_set_list=[]
 
    path=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_sets_ambient_test/*.pq')]
    print(path)
    for j in path:
        fold_set_list+=glob.glob(j)
    
    fold=pd.read_parquet(fold_set_list)
    #X_, y_, name_list, times=fs_to_Xy_horizontal(fold, params)
   
    
    X_, y_, name_list, times, snr_metric=fs_to_Xy(fold, param, test)


    return X_, y_, name_list, times, snr_metric


def fs_to_Xy(df, param, test=True):
    
    '''
    function to horizontally concatenate velocity components (East, North Up) into feature sets.
    Feature vector m samples x 3*n (features).
    Target vector is a single m x 1, with logic to check each component's label into single label.
    
    "Maybes" are values I labeled as maybe motion.  Maybe variable maps these into 1 (no motion) or 2 (motion)
    '''
    X_list=[]
    y_list=[]
    threshold_list=[]
    name_list_tmp=[]
    

    for i,direc in enumerate(param['dims']):
        psd_tmp=pd.DataFrame(df['%s_psd' %direc].tolist(), index= df.index).to_numpy()
        psd_names=['%s_P%s' %(direc,f) for f in np.arange(psd_tmp.shape[1])]
        
        time_tmp=pd.DataFrame(df['%s_time' %direc].tolist(), index= df.index).to_numpy()
        time_names=['max1','max2','max3','max4','med','mad']
        
        wl_tmp=pd.DataFrame(df['%s_wl_pk' %direc].tolist(), index= df.index).to_numpy()
        wl_sum_tmp=pd.DataFrame(df['%s_wl_sum' %direc].tolist(), index= df.index).to_numpy()
        wl_names=['%s_wl%s' %(direc,f) for f in np.arange(wl_tmp.shape[1])]
        
        
        if param['feature'] == 'psd':
            X_temp=np.hstack((time_tmp,psd_tmp))
            names=time_names+psd_names
            
        if param['feature'] == 'wavelet':
            X_temp=np.hstack((time_tmp,wl_tmp))
            names=time_names+wl_names
        if param['feature'] == 'time':
            X_temp=time_tmp
            names=time_names
        if param['feature'] == 'all':
            X_temp=np.hstack((time_tmp,psd_tmp, wl_tmp))
            names=time_names+psd_names+wl_names
                    

        X_list+=[X_temp]
        name_list_tmp+=[names]
        
        #########TARGET VECTOR#########
        y_temp=df['%s_Y' %direc]  
        y_list+=[y_temp]
        
        ########threshold vector#######
        threshold_tmp=df['%s_wl_snr' %direc]  
        threshold_tmp=df['%s_psd_snr' %direc]
        threshold_list+=[threshold_tmp]
        
    X=np.hstack(X_list)
    name_list=np.hstack(name_list_tmp)
    y_tmp=np.vstack(y_list)
    
    #logic : if any component has motion, say motion is present
    y=np.sum(y_tmp, axis=0)
    y=np.where(y>0, 1, y)
    
    #In training, filter by wl threshold
    if test is False:
        
        thresh_=np.vstack(threshold_list)
        threshold=np.max(thresh_, axis=0)
        
        # if y>0 and threshold < value, drop
        mask=~((threshold<param['wl_thresh'])&(y>0))

        
        X=X[mask]
        y=y[mask]
    
    snr_metric=df[['H0_psd_snr','H0_psd_snr','H0_psd_snr']].max(axis=1)


                                                   
    return X, y, name_list, df.time, snr_metric