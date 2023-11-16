import os
import glob

import numpy as np
import pandas as pd
import s3fs

from multiprocessing import Pool, cpu_count
from generate_ts_utils import *


class vel_ts:
    '''
    A class to represent a three-component velocity timeseries.

    ...

    Attributes
    ----------
    array : list
        ngaw2 event station array 
        'Record Sequence Number','Earthquake Name','Year','Station Name','Magnitude','Mechanism','Rjb (km)',
            'Rrup (km)','Vs30 (m/sec)'
    record_number : str
        ngaw2 record number
    age : int
        age of the person

    Methods
    -------
    vel_load_ds(self, target_sr):
        loads, downsamples and labels NGAW2 waveforms. 
    '''
    def __init__(self, array):
        self.array = array
        self.record_number= self.array[0]
        
    def vel_load_ds(self, target_sr):
        '''
        loads, downsamples and labels NGAW2 waveforms.
        '''
        s3 = s3fs.S3FileSystem(anon=False)
        s3_prefix='s3://'
        bucket='gnss-ml-dev-us-east-2-gbmm8xhl6lon'
        key='ngaw2/RSN%s_*.VT2' %self.record_number
        sm_fn=s3.glob(s3_prefix+bucket+'/'+key)
        
        sm_fn=[s3_prefix + s for s in sm_fn]
        
        scale_f, dt= meta_check(sm_fn[0])

        # determine vertical TS
        up_idx=vertical_idx(sm_fn)

        #sort out components- distinguish between Vert and Horiz for gnss noise
        h_idxs=np.delete(np.arange(3),up_idx)
        h0_idx=h_idxs[0]
        h1_idx=h_idxs[1]
        
        sr=1/dt
        deci=int(sr/target_sr)
        
        self.H0, self.time, self.H0_y=load_ds_label(sm_fn[h0_idx],sr, deci, buffer45=True)
        self.H1, self.time, self.H1_y=load_ds_label(sm_fn[h1_idx],sr, deci, buffer45=True)
        self.UP, self.time, self.UP_y=load_ds_label(sm_fn[up_idx],sr, deci, buffer45=True)
        
        useable_len=min(len(self.H1),len(self.H0),len(self.UP))
        
        self.H0, self.time, self.H0_y=self.H0[:useable_len], self.time[:useable_len], self.H0_y[:useable_len]
        self.H1, self.time, self.H1_y=self.H1[:useable_len], self.time[:useable_len], self.H1_y[:useable_len]
        self.UP, self.time, self.UP_y=self.UP[:useable_len], self.time[:useable_len], self.UP_y[:useable_len]
        
def generate_gnss_ts(nga_event_station):
    '''
    generate psuedo synthetic gnss velocity timeseries from ngaw2 waveforms at 7 noise levels
    saves timeseries to parquet files

    inputs:
    -------
    nga_event_station: list
        list of station + metadata
    '''
    try:
        vel_1=vel_ts(nga_event_station)
        target_sr=5
        vel_1.vel_load_ds(target_sr)

        for noise_lev in np.arange(5,100,15):
            store_df_li=[]
            samples2add=np.random.randint(0,60*5)
            for name,ts,ts_labels in zip(['H0','H1','UP'],[vel_1.H0,vel_1.H1,vel_1.UP],[vel_1.H0_y,vel_1.H1_y,vel_1.UP_y]):
                #add random buffer
                ts,t=add_random_buffer(ts, vel_1.time, samples2add,target_sr)
                ts_labels=np.concatenate((np.zeros(samples2add) ,ts_labels))
                noise_ts=create_noise_ts(n_steps=len(ts), level=noise_lev, name=name)
                combined=ts+noise_ts
                store_df=pd.DataFrame(np.vstack((t,combined, noise_ts, ts_labels.astype(int))).transpose(), 
                                columns=['t','%s_ts' %name,'%s_noise'%name, '%s_label' %name])

                store_df.set_index('t', inplace=True)
                store_df_li.append(store_df)
            write_to_pq(store_df_li, vel_1.array, noise_lev)
    except Exception as e:
        print(e.args)
        print(nga_event_station)
        pass


def mp_handler():
    
    nga_event_station_list=create_nga_event_station_list()
    print(len(nga_event_station_list))
    #nga_event_station_list=nga_event_station_list[:1]
    #print(nga_event_station_list)
    
    #initiate pool to parallel process stations
    print("There are {} CPUs on this machine ".format(cpu_count()))
    pool = Pool(cpu_count()-2)
    pool.map(generate_gnss_ts, nga_event_station_list)
    pool.close()

if __name__ == '__main__':
    mp_handler()