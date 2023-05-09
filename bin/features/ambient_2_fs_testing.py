import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import glob
import random

import numpy as np
import os
import pandas as pd
import numpy as np

import datetime
from scipy import signal
import pywt

from multiprocessing import Pool, cpu_count

#import sys
#sys.path.insert(1, '/home/ec2-user/pgv_ml/snivel/notebooks/')
from pgv_ml_utils import *


def get_features_ambient(tmp,obs):
    
    #TIME
    max_arr=tmp.abs().nlargest(4).values
    med_arr=tmp.median()
    #mad_arr=tmp.mad()
    mad_arr=np.abs(tmp-tmp.median()).median()
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

    y_label=0
        
        
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

def write_to_pq_ambient(store_df_li, meta_array, noise_lev):
    ts_df=pd.concat(store_df_li, axis=0)
    ts_df=ts_df.reset_index(drop=True)
    
    meta_dict=make_meta_dict(meta_array, noise_lev)
    print(meta_dict)

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
    # CHANGE THIS FOR UNSEEN vs TRAINING
    #fpath=os.path.join(path.parent.absolute(),'data','feature_sets','%s_%sa.pq' %(meta_dict['record_number'],meta_dict['station'])) 
    fpath=os.path.join(path.parent.absolute(),'data','feature_sets_ambient_test','%s_%sa.pq' %(meta_dict['record_number'],meta_dict['station']))
    pq.write_table(table, fpath)



def ambient_2_pq(file,set_num):
    try:
        window_size=30
        obs_list=['dedt', 'dndt', 'dudt']
        direc=['H0','H1','UP']


        station=file[-17:-13]
        year=file[-8:-4]
        doy=file[-12:-9]


        df=read_vel_fname(file)
        # Only use the first 30 minutes
        ### EDIT HERE
        
        #df=df[:int(30*60*5)]
        # FOR UNSEEN TESTING USE 30:
        df=df[int(30*60*5):]
        
        feature_list=[]

        num_win=(len(df)-len(df)%(window_size*5))/((window_size/3)*5)

        win_start=df.index[0]
        z=0
        while win_start < (df.index[-1] - datetime.timedelta(seconds=window_size)):
            win_stop=win_start+datetime.timedelta(seconds=window_size)
            temp_df=df.loc[win_start:win_stop]

            tmp_list=[]
            for i,obs in enumerate(obs_list):
                #plotting

                #features
                features=get_features_ambient(temp_df[obs], obs)
                tmp_list.append(features)
            #axes[3].plt.plot(features.dudt_power.values[0])
            feature_list.append(tmp_list[0].merge(tmp_list[1], on='time').merge(tmp_list[2], on='time'))
            #win_start+=datetime.timedelta(seconds=window_size/3)
            #Ambient noise doesn't benefit from overlapping
            win_start+=datetime.timedelta(seconds=window_size)
            z+=1

        noise_lev=0
        meta_array=9*[np.nan]
        #station = 
        meta_array[3]=station
        #year
        meta_array[2]=year
        #rsn = doy+year
        meta_array[0]=str(int(year)*1000+int(doy))
        #event == ambient set num
        meta_array[1]='ambient_'+str(set_num)


        write_to_pq_ambient(feature_list, meta_array, noise_lev)
    except Exception as e:
        print(e)
        pass

    
def mp_handler():
  
    path = '../data/jgr_data/output/ambient_set_2/velocities*'
    list_files=glob.glob(path)
    #list_files=glob.glob(path)[:150]
    random.shuffle(list_files)

    number_sets=50
    set_divisor=int(len(list_files)/number_sets)+1
    
    set_num=np.arange(len(list_files))//set_divisor
    
    #write function to check whats already written and not reprocess
    # read metadata of existing pq store to compile list of events-stations
    # difference two lists for residual to process
    
    #initiate pool to parallel process stations
    print("There are {} CPUs on this machine ".format(cpu_count()))
    #pool = Pool(cpu_count()-2)
    pool=Pool(4)
    pool.starmap(ambient_2_pq, zip(list_files,set_num))
    pool.close()

if __name__ == '__main__':
    mp_handler()