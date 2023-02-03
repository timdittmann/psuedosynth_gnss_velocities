#
#imports jgr feature sets and maps them to new featuresets
# old was max, psd bins
# new is new time domain, psd and wavelets

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import glob
import random


import numpy as np

from itertools import product
import os
import pandas as pd
import numpy as np
import glob

import datetime
from scipy import signal
import pywt

import pyarrow as pa
import pyarrow.parquet as pq
import json

from multiprocessing import Pool, cpu_count

import sys
sys.path.insert(1, '/home/ec2-user/pgv_ml/snivel/notebooks/')
#from pgv_ml_utils import *



def get_features_old(tmp,obs, label):
    
    #TIME
    max_arr=tmp.abs().nlargest(4).values
    med_arr=tmp.median()
    mad_arr=tmp.mad()
    time_f=np.concatenate((max_arr,[med_arr],[mad_arr]))

    #frequency
    f, p=signal.periodogram(x=tmp.values, fs=5, nfft=5*30)

    #time-FREQUENCY
    per_min=.4
    per_max=10
    f_min = 1/per_max
    f_max = 1/per_min
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    w0=1
    scales=w0 / (2 * np.pi * f)

    [coefficients, frequencies] = pywt.cwt(tmp, scales, 'morl', 5)
    wl_pk=abs(coefficients).max(axis=1)
    wl_sum=abs(coefficients).sum(axis=1)

    wl_snr=np.nan
    snr_db=np.nan

    if ((label==1) | (label==3)):
        y_label=0
    if label==2:
        y_label=1
        
        
    if obs == 'dedt':
        direc='H0'
    if obs == 'dndt':
        direc='H1'
    if obs == 'dudt':
        direc='UP'


    feature_ds = pd.DataFrame({'time':[pd.to_datetime(tmp.index[-1]).value/ 10**9], direc+'_time':[time_f], 
                               direc+'_psd':[p], direc+'_psd_snr':[snr_db], direc+'_wl_pk':[wl_pk], 
                               direc+'_wl_sum':[wl_sum],
                   direc+'_wl_snr':[wl_snr], direc+'_Y_sum':[float(0)], direc+'_Y':[y_label]})
    '''
    feature_ds = pd.DataFrame({'time':[tmp.index[-1].astype(int)], direc+'_time':[time_f], 
                               direc+'_psd':[p], direc+'_psd_snr':[snr_db], direc+'_wl_pk':[wl_pk], direc+'_wl_sum':[wl_sum],
                   direc+'_wl_snr':[wl_snr], direc+'_Y_sum':[0], direc+'_Y':[y_label]})
    '''

    return feature_ds



def make_meta_dict(meta_array, noise_lev):
    meta_content = {
            "station":meta_array[3],
            "year":meta_array[2],
            "eq_name":meta_array[1],
            "record_number":meta_array[0],
            "magnitude":meta_array[4],
            "mechanism":meta_array[5],
            "Rjb":meta_array[6],
            "Rrup":meta_array[7],
            "Vs30": meta_array[8],
            "noise_lev": str(noise_lev),
        }
    return meta_content

def write_to_pq_old2new(store_df_li, meta_array, noise_lev):
    ts_df=pd.concat(store_df_li, axis=1)
    ts_df = ts_df.loc[:,~ts_df.columns.duplicated()]
    ts_df=ts_df.reset_index(drop=True)
    
    meta_dict=make_meta_dict(meta_array, noise_lev)

    meta_key = 'feature_meta'

    table = pa.Table.from_pandas(ts_df)

    meta_json = json.dumps(meta_dict)

    existing_meta = table.schema.metadata
    combined_meta = {
        meta_key.encode() : meta_json.encode(),
        **existing_meta
    }

    table = table.replace_schema_metadata(combined_meta)
    from pathlib import Path
    path = Path(os.getcwd())
    fpath=os.path.join(path.parent.absolute(),'data','feature_sets_JGR','%s' %(meta_dict['record_number'])) 
    pq.write_table(table, fpath)





import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import glob
import random



def oldfs_2_newfs(infile):
    try:
        obs_list=['dedt', 'dndt', 'dudt']
        direc=['H0','H1','UP']
        tmp=pd.read_parquet(infile)
        
        feature_list=[]
        
        for i,obs in enumerate(obs_list):
            tmp_list=[]

            for i, row in pd.DataFrame(tmp['%s_raw' %obs].to_list()).iterrows():
                label=tmp.iloc[i]['%s_Y' %obs]

            #features
                features=get_features_old(row, obs, label)
                tmp_list.append(features)
            feature_list.append(pd.concat(tmp_list))

        noise_lev=0
        meta_array=9*[np.nan]
        #station = 
        meta_array[3]=''
        #year
        meta_array[2]=''
        #rsn = doy+year
        meta_array[0]=os.path.split(infile)[-1]
        #event == ambient set num
        meta_array[1]=''


        write_to_pq_old2new(feature_list, meta_array, noise_lev)
    except Exception as e:
        print(e)
        pass

    
def mp_handler():
  
    path = '/home/ec2-user/pgv_ml/snivel/data/feature_set/*'

    list_files=glob.glob(path)

    
    #initiate pool to parallel process stations
    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(cpu_count()-2)
    pool.map(oldfs_2_newfs, list_files)
    pool.close()

if __name__ == '__main__':
    mp_handler()