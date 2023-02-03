import pandas as pd
from obspy import read
import obspy
from obspy.signal import PPSD
from numpy import where,mean
from glob import glob
from matplotlib import pyplot as plt

import numpy as np

from obspy import read
from pathlib import Path


from noise_utils import psd_stack2hist_stack, _plot_histogram, get_percentile


#pq_file='/home/ec2-user/pgv_ml/synth_gnss_vel/data/ambient_5sps_6yr.pq'
pq_file='../data/jgr_data/ambient_5sps_6yr.pq'
df3=pd.read_parquet(pq_file)

sac_dir='../data/ambient_sac/'

H_val=[]
V_val=[]
 
db_bins=(-70, 0, 0.5)
ppsd_length=3600/3
period_limits=[2/5,600]

for k in np.arange(len(df3)):
#for k in np.arange(1):
    try:
        site=df3.loc[k].station.upper()
        doy=df3.loc[k].doy
        year=df3.loc[k].year
   
        e = read(sac_dir+site.upper()  + '_' + doy + '_' + year +  '.vel.e', debug_headers=True) 
        n = read(sac_dir+site.upper()  + '_' + doy + '_' + year +  '.vel.n', debug_headers=True)
        z = read(sac_dir+site.upper()  + '_' + doy + '_' + year +  '.vel.u', debug_headers=True)

        paz = {'gain': 1,'poles': [1],'sensitivity': 1,'zeros': [0j, 0j]}

        Eppsd = PPSD(e[0].stats, paz, db_bins=db_bins,ppsd_length=ppsd_length, special_handling="ringlaser")
        Nppsd = PPSD(n[0].stats, paz, db_bins=db_bins,ppsd_length=ppsd_length, special_handling="ringlaser")
        Zppsd = PPSD(z[0].stats, paz, db_bins=db_bins,ppsd_length=ppsd_length, special_handling="ringlaser")

        #Add traces to ppsd object
        Eppsd.add(e) 
        Nppsd.add(n) 
        Zppsd.add(z) 
        H_val+=Eppsd.psd_values
        H_val+=Nppsd.psd_values
        V_val+=Zppsd.psd_values

    except:
        pass

H_stacked=np.vstack(H_val)
V_stacked=np.vstack(V_val)

'''
np.save('models/e_stacked_ambient.npy', e_stacked) 
np.save('models/n_stacked_ambient.npy', n_stacked) 
np.save('models/z_stacked_ambient.npy', z_stacked) 

Eppsd.save_npz("models/Eppsd.npz") 
Nppsd.save_npz("models/Nppsd.npz")
Zppsd.save_npz("models/Zppsd.npz")

'''


for psd_stack_array,name in zip([H_stacked,V_stacked],["H","V"]):
    hist_stack, hist_stack_cumul=psd_stack2hist_stack(psd_stack_array, Eppsd)
    filename='figs/ambientnoise_%s.png' %name
    fig, _db_bin_edges, period_bin_centers, percentiles=_plot_histogram(Eppsd, hist_stack,hist_stack_cumul, db_bins, draw=False, filename=filename)
    
    # save to array
    per_li=[]
    for i,percentile in enumerate(np.arange(5,100,5)):
        periods, percentile_values=get_percentile(hist_stack_cumul, percentile, _db_bin_edges, period_bin_centers)
        if i == 0:
            per_li.append(periods)
        per_li.append(percentile_values)
    per_array=pd.DataFrame(per_li)
    new_header = per_array.iloc[0] #grab the first row for the header
    per_array = per_array[1:] #take the data less the header row
    per_array.columns = new_header #set the header row as the df header
    per_array['percentile']=np.arange(5,100,5)
    per_array=per_array.set_index('percentile')
    per_array.to_csv("../data/%s_psd_percentiles.csv" %name)

