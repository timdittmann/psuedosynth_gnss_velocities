#for index, row in meta[:4].iterrows():
# Better still, make a queue of record numbs than MultiProcess?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from obspy.signal.filter import lowpass
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import json


def create_nga_event_station_list(rsn=False):
    '''
    '''
    
    if os.path.basename(os.getcwd()) == 'mkfigs':
        root_dir='../../'
    else:
        root_dir='../'
        
    path = Path(os.getcwd())
    meta_directory=os.path.join(root_dir,'data','ngaw2','*.csv')
    
    meta_files=glob.glob(meta_directory)

    meta_list=[]
    for meta_file in meta_files:
        meta_tmp=pd.read_csv(meta_file, sep=',',header=28,skipfooter=235,engine='python')
        meta_tmp=meta_tmp.drop(columns=['Result ID'])
        meta_list.append(meta_tmp)
    meta=pd.concat(meta_list, ignore_index=True) 
    meta.columns = meta.columns.str.strip()
    meta=meta.drop_duplicates(subset=['Record Sequence Number'], ignore_index=True)    
    columns=['Record Sequence Number','Earthquake Name','Year','Station Name','Magnitude','Mechanism','Rjb (km)',
            'Rrup (km)','Vs30 (m/sec)']
    if rsn:
        meta=meta[meta['Record Sequence Number']==rsn]
    return meta[columns].values.tolist()

def load_downsample(fname, sr, deci, buffer45):
    sm_vel=pd.read_csv(fname, header=None,skiprows=4, delim_whitespace=True).dropna()
    y=sm_vel.values.flatten()/100 #cm to meters
    times=np.arange(len(y))/sr

    # downsample after lowpassing the signal
    y_l = lowpass(y, 2.5, df=sr, corners=4, zerophase=False)
    y_new = y_l[::deci]
    t_new=times[::deci]

    if buffer45 is True:
        y_new, t_new=add_random_buffer(y_new,t_new,45*5,sr/deci)
            
    return y_new, t_new

def load_ds_label(sm_f,sr, deci, buffer45=True):
    y_new, t_new = load_downsample(sm_f,sr, deci, buffer45=True)
    
    label_arr, first_t, last_t=stalta_labels(t_new, y_new, sample_rate=sr/deci)
    return y_new, t_new, label_arr
    
    
def meta_check(vel_file):
    # convert to m/s, default from cm/s, but will check
    scale_f=100

    with open(vel_file, 'rb') as infile:
        for lineno, line in enumerate(infile): 
            if line.decode("utf-8").startswith('VELOCITY'):
                units=((line.split()[6]))
                if units.decode("utf-8") == 'CM/S':
                    scale_f=100
                if units.decode("utf-8") == 'M/S':
                    scale_f=1
                if units.decode("utf-8") == 'MM/S':
                    scale_f=1000
            if line.decode("utf-8").startswith('NPTS'):
                dt=float((line.split()[3][:6]))
                break
    return scale_f, dt

def stalta_labels(time_array, ts_array, sample_rate):
    '''
    '''
    label_arr=np.zeros(len(ts_array))

    from obspy.signal.trigger import recursive_sta_lta
    from obspy.signal.trigger import classic_sta_lta

    cft = recursive_sta_lta(ts_array, int(5 * sample_rate), int(10 * sample_rate))
    #plt.plot(time_array,cft)
    #buffer length
    buffer_l=len(cft[time_array<=0])
    
    #find first trigger sample after OT
    first=np.argmax(cft[time_array>0]>1.75)+buffer_l
    #determine time of sample
    first_t=time_array[first]
    #logic for end of sta/lta not being included
    if np.argmax(cft[first:]<0.5) == 0:
        last=np.nan
        last_t=np.nan
        label_arr[first:]=1
    else:
        last=np.argmax(cft[first:]<0.5)+first
        last_t=time_array[last]
        label_arr[first:last]=1
    return label_arr, first_t, last_t

def vertical_idx(ts_list):
    for i in np.arange(len(ts_list)):
        if ((ts_list[i][-5:-4]=='V')|(ts_list[i][-6:-4]=='UP')|(ts_list[i][-6:-4]=='DN')
            |(ts_list[i][-6:-4]=='WN')|(ts_list[i][-6:-4]=='UD')|(ts_list[i][-5:-4]=='Z')):
            up_idx=i
        else:
            up_idx=2
    return up_idx

def add_random_buffer(y,t,samples,target_sr):
    y_add=np.random.normal(0,.01/100,samples)
    t_add=np.arange(-len(y_add)+t.min(),+t.min(),1)/target_sr
    y_new2=np.concatenate((y_add,y))
    t_new2=np.concatenate((t_add,t))
    return y_new2, t_new2

def create_noise_ts(n_steps, level, name):
    if name == 'UP':
        psd_df=get_psd("V")
    else:
        psd_df=get_psd("H")

    ppsd_out, frequencies=modify_for_noise_df(psd_df, level=level)
    noise_ts=make_noise(n_steps,frequencies,ppsd_out,PGD=True)
    return noise_ts

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

def write_to_pq(store_df_li, meta_array, noise_lev):
    ts_df=pd.concat(store_df_li, axis=1)
    
    meta_dict=make_meta_dict(meta_array, noise_lev)
    #print(meta_dict)

    meta_key = 'feature_meta'

    table = pa.Table.from_pandas(ts_df)

    meta_json = json.dumps(meta_dict)

    existing_meta = table.schema.metadata
    combined_meta = {
        meta_key.encode() : meta_json.encode(),
        **existing_meta
    }

    table = table.replace_schema_metadata(combined_meta)
    path = Path(os.getcwd())
    fpath=os.path.join(path.parent.absolute(),'data','synth_ts','%s_%02d.pq' %(meta_dict['record_number'],noise_lev)) 
    pq.write_table(table, fpath)
    
    
###############################

def windowed_gaussian(duration,hf_dt,window_type='saragoni_hart',M=5.0,dist_in_km=50.,std=1.0,ptime=10,stime=20):
    '''
    Get a gaussian white noise time series and window it
    modified from D.Melgar Mudpy
    '''
    
    from numpy.random import normal
    from numpy import log,exp,arange
    from scipy.special import gamma
    
    mean=0.0
    num_samples = int(duration/hf_dt)
    #If num_smaples is even then make odd for FFT stuff later on
    if num_samples%2==0:
        num_samples+=1
    t=arange(0,num_samples*hf_dt,hf_dt)
    t=t[0:num_samples]

    noise = normal(mean, std, size=num_samples)
    
    if window_type=='saragoni_hart':
        epsilon=0.2
        eta=0.05
        b=-epsilon*log(eta)/(1+eta*(log(epsilon)-1))
        c=b/(epsilon*duration)
        a=(((2*c)**(2*b+1))/gamma(2*b+1))**0.5
        window=a*t**b*exp(-c*t)
    elif window_type=='cua':
        ptime=0
        window=cua_envelope(M,dist_in_km,t,ptime,stime,Pcoeff=0,Scoeff=12)
    elif window_type==None: #jsut white noise, no window
        window=1
        
        
    noise=noise*window
    #plt.plot(noise)
    return noise

        
def apply_spectrum(w,A,f,hf_dt,is_gnss=False,gnss_scale=(1/2**0.5)*.5): #.5
    '''
    Apply the modeled spectrum to the windowed time series
    modified from D.Melgar Mudpy
    
    '''
   
    from numpy import fft,angle,cos,sin,mean,zeros,real
    from scipy.interpolate import interp1d
    
    #to frequency domain
    fourier=fft.fft(w)
    freq=fft.fftfreq(len(w),hf_dt)
    
    #Get positive frequencies
    Nf=len(freq)
    positive_freq=freq[1:int(1+Nf/2)]
      
    #Make POWER spectrum of windowed time series have a mean of 1
    #norm_factor=hf_dt*mean(abs(fourier)**2)**0.5
    #norm_factor=mean(abs(fourier))
    norm_factor=mean(abs(fourier)**2)**0.5
    fourier=fourier/norm_factor
    
    #Keep phase
    phase=angle(fourier)
    
    #resample model amplitude spectr to frequencies
    interp=interp1d(f,A,bounds_error=False)
    amplitude_positive=interp(positive_freq)
    
    #Place in correct order A[0] is DC value then icnreasing positive freq then decreasing negative freq
    amplitude=zeros(len(freq))
    #DC value
    amplitude[0]=0
    #Positive freq. div by 2 to keep right power
    amplitude[1:int(1+Nf/2)]=amplitude_positive/2
    #Negative freq
    amplitude[int(1+Nf/2):]=amplitude_positive[::-1]/2
    
    #Scale for GNSS displacememnts?
    if is_gnss:
        #amplitude
        #amplitude /= gnss_scale 
        amplitude /= (gnss_scale) 
        
    #Apply model amplitude spectrum
    amplitude=amplitude*abs(fourier)
    
    #Obtain complex foureier series
    R=amplitude*cos(phase)
    I=amplitude*sin(phase)
    fourier=R+I*1j
    
    #ifft
    seis=real(fft.ifft(fourier))
    
    
    if is_gnss:
        seis *= len(seis)**0.5
        
    else:
        seis=seis*len(seis)
    
    return seis         

def make_noise(n_steps,f,epsd,PGD=False):
    '''
    modified from D.Melgar Mudpy
    '''
    #define sample rate
    dt=0.2 #make this 1 so that the length of PGD can be controlled by duration
    
    duration=n_steps
    # get white noise
    E_noise=windowed_gaussian(duration,dt,window_type=None)
    #N_noise=windowed_gaussian(duration,dt,window_type=None)
    #Z_noise=windowed_gaussian(duration,dt,window_type=None)
    noise=windowed_gaussian(duration,dt,window_type=None)
    #get PSDs
    #f,Epsd,Npsd,Zpsd=gnss_psd(level=level,return_as_frequencies=True,return_as_db=False)
    #control the noise level
    #scale=np.abs(np.random.randn()) #normal distribution
    #Epsd=Epsd*scale
    #Npsd=Npsd*scale
    #Npsd=Npsd*scale
    #Covnert PSDs to amplitude spectrum
    epsd = (epsd)**0.5
    #Npsd = Npsd**0.5
    #Zpsd = Zpsd**0.5
    #apply the spectrum
    E_noise=apply_spectrum(E_noise,epsd,f,dt,is_gnss=PGD)[:n_steps]
    E_noise=np.real(E_noise)
    #N_noise=apply_spectrum(N_noise,Npsd,f,dt,is_gnss=True)[:n_steps]
    #Z_noise=apply_spectrum(Z_noise,Zpsd,f,dt,is_gnss=True)[:n_steps]
    return E_noise


def get_psd(h_or_v="H"):
    if os.path.basename(os.getcwd()) == 'mkfigs':
        root_dir='../../'
    else:
        root_dir='../'
        
    psd_df=pd.read_csv(root_dir+"models/%s_psd_percentiles.csv" %h_or_v)
    return psd_df

def modify_for_noise_df(psd_df, level=50):
    from numpy import r_
    frequencies=1/np.array(psd_df.columns.values.tolist()[1:]).astype(float)[::-1]

    #reverse psds
    #ppsd_out=ppsd_in[1][::-1]
    ppsd_out=psd_df[psd_df.percentile==level].values[0][1:][::-1]
    #linearize
    ppsd_out=10**(ppsd_out/10)
    #add zero frequency
    frequencies=r_[0,frequencies]

    #add zero frequency values
    ppsd_out=r_[ppsd_out[0],ppsd_out] 
    return ppsd_out, frequencies