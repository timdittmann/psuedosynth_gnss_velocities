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


'''
def get_features(temp_df,direc):
    tmp=temp_df['%s_ts' %direc]
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

    #SNR metric
    tmp_noise=temp_df['%s_noise' %direc]
    tmp_sig=tmp-tmp_noise
    [sig_coefficients, frequencies] = pywt.cwt(tmp_sig, scales, 'morl', 5)
    [noise_coefficients, frequencies] = pywt.cwt(tmp_noise, scales, 'morl', 5)
    wl_snr=np.max((np.abs(sig_coefficients)-np.abs(noise_coefficients)))

    y_label=0
    if temp_df['%s_label' %direc].sum()>0:
        y_label=1

    feature_ds = pd.DataFrame({'time':[tmp.index[-1]], direc+'_time':[time_f], 
                               direc+'_psd':[p], direc+'_wl':[coefficients.flatten()],
                   direc+'_wl_snr':[wl_snr], direc+'_Y_sum':[temp_df['%s_label' %direc].sum()], direc+'_Y':[y_label]})

    return feature_ds
'''

def ts_features(tmp):
    max_arr=tmp.abs().nlargest(4).values
    med_arr=tmp.median()
    #mad_arr=tmp.mad()
    mad_arr=np.abs(tmp-tmp.median()).median()
    time_f=np.concatenate((max_arr,[med_arr],[mad_arr]))
    return time_f

def freq_features(tmp):
    f, p=signal.periodogram(x=tmp.values, fs=5, nfft=5*30)
    return f,p

def time_freq_features(tmp):
    per_min=.4
    per_max=10
    f_min = 1/per_max
    f_max = 1/per_min
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    w0=1
    scales=w0 / (2 * np.pi * f)

    [coefficients, frequencies] = pywt.cwt(tmp, scales, 'morl', 5)
   
    return [coefficients, frequencies]
    
    
def get_features(temp_df,direc):
    tmp=temp_df['%s_ts' %direc]
    #TIME
    time_f=ts_features(tmp)

    #frequency
    f, p = freq_features(tmp)
    
    #time-FREQUENCY
    [coefficients, frequencies]=time_freq_features(tmp)
    wl_pk=abs(coefficients).max(axis=1)
    wl_sum=abs(coefficients).sum(axis=1)

    #SNR metric
    tmp_noise=temp_df['%s_noise' %direc]
    tmp_sig=tmp-tmp_noise
    
    per_min=.4
    per_max=10
    f_min = 1/per_max
    f_max = 1/per_min
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    w0=1
    scales=w0 / (2 * np.pi * f)
    
    [sig_coefficients, frequencies] = pywt.cwt(tmp_sig, scales, 'morl', 5)
    [noise_coefficients, frequencies] = pywt.cwt(tmp_noise, scales, 'morl', 5)
    wl_snr=np.max((np.abs(sig_coefficients)-np.abs(noise_coefficients)))
    
    f_sig,p_sig=signal.periodogram(tmp_sig,fs=5)
    f_noi,p_noi=signal.periodogram(tmp_noise,fs=5)
    
    snr_db=20*np.log10(p_sig[1:]/p_noi[1:]).max()
    

    y_label=0
    if temp_df['%s_label' %direc].sum()>0:
        y_label=1

    feature_ds = pd.DataFrame({'time':[tmp.index[-1]], direc+'_time':[time_f], 
                               direc+'_psd':[p], direc+'_psd_snr':[snr_db], direc+'_wl_pk':[wl_pk], direc+'_wl_sum':[wl_sum],
                   direc+'_wl_snr':[wl_snr], direc+'_Y_sum':[temp_df['%s_label' %direc].sum()], direc+'_Y':[y_label]})

    return feature_ds


def read_meta(path):
    custom_meta_key = 'feature_meta'
    table = pq.read_table(path)
    meta_json = table.schema.metadata[custom_meta_key.encode()]
    meta = json.loads(meta_json)
    return meta

def ts_2_feature_file(file):
    window_size=30*5
    step_=5*5 #run every 10 second
    obs_list=['H0','H1','UP']
    feature_list=[]
    
    ts=pd.read_parquet(file)
    win_start=0
    while win_start < (len(ts) - window_size):
        win_stop=win_start+window_size
        temp_df=ts.iloc[win_start:win_stop]
        tmp_list=[]
        for direc in obs_list:
            features=get_features(temp_df,direc)
            tmp_list.append(features)
        feature_list.append(tmp_list[0].merge(tmp_list[1], on='time').merge(tmp_list[2], on='time'))    
        win_start+=step_
        
    feature_df=pd.concat(feature_list)
    feature_df=feature_df.reset_index(drop=True)

    table = pa.Table.from_pandas(feature_df)

    custom_meta_key = 'feature_meta'
    custom_meta_json = json.dumps(read_meta(file))

    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode() : custom_meta_json.encode(),
        **existing_meta
    }

    table = table.replace_schema_metadata(combined_meta)
    
    fn=os.path.basename(file)
    from pathlib import Path
    path = Path(os.getcwd())
    fpath=os.path.join(path.parent.absolute(),'data','feature_sets',fn)
    
    pq.write_table(table, fpath)
    
def mp_handler():
    
    from pathlib import Path
    path = Path(os.getcwd())
    fpath=os.path.join(path.parent.absolute(),'data','synth_ts') 

    file_list=[]
    for file in os.listdir(fpath):
        if file.endswith(".pq"):
            file_list+=[os.path.join(fpath, file)]
            
    print(file_list[0])
    
    #write function to check whats already written and not reprocess
    # read metadata of existing pq store to compile list of events-stations
    # difference two lists for residual to process
    
    #initiate pool to parallel process stations
    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(cpu_count()-2)
    pool.map(ts_2_feature_file, file_list)
    pool.close()

if __name__ == '__main__':
    mp_handler()