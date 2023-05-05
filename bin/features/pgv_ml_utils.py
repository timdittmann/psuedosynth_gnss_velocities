#!/usr/bin/env python

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import pyarrow.parquet as pq
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import precision_recall_curve

# optimal threshold for precision-recall curve with logistic regression model
from numpy import argmax

from collections import deque
from scipy import signal

import urllib.request

def read_meta(path):
    custom_meta_key = 'feature_meta'
    table = pq.read_table(path)
    meta_json = table.schema.metadata[custom_meta_key.encode()]
    meta = json.loads(meta_json)
    return meta

# FEATURE SET FUNCTIONS
pd.options.mode.chained_assignment = None  # default='warn'
def fs_to_Xy_horizontal(df, param):
    
    '''
    function to horizontally concatenate velocity components (East, North Up) into feature sets.
    Feature vector m samples x 3*n (features).
    Target vector is a single m x 1, with logic to check each component's label into single label.
    
    "Maybes" are values I labeled as maybe motion.  Maybe variable maps these into 1 (no motion) or 2 (motion)
    '''
    X_list=[]
    y_list=[]
    name_list_tmp=[]
    
    # drop 3's
    # logic, if row contains a 3 but not a 1 
    #if fs_params['maybes']==3:
        #df=df[(df.dedt_Y!=3) & (df.dndt_Y!=3) & (df.dudt_Y!=3)]
    df=df.drop(df[((df.dedt_Y==3) | (df.dndt_Y==3) | (df.dudt_Y==3)) & ((df.dedt_Y!=2) & (df.dndt_Y!=2) & (df.dudt_Y!=2))].index)
    #vertically stack components into single arrays (X, Y)

    for direc in param['dims']:
        ########## FEATURE VECTOR ############
        #pick features of interest
        columns=['d%sdt_max_val' %direc, 'd%sdt_max2_val' %direc, 'd%sdt_med_val' %direc, 
                 'd%sdt_min_val' %direc, 'd%sdt_min2_val' %direc, 'd%sdt_mad_val' %direc]
        ##
        columns2=['d%sdt_max_val' %direc, 'd%sdt_max2_val' %direc,  'd%sdt_min_val' %direc, 'd%sdt_min2_val' %direc]
        
        # Modify orignal max/min features:
        # take abs of  4 max/min, then calculate snr
        # add median
        # add mad
        # add snr metric (median of 4 highest / noise threshold (median +3* mad)
       
        temp=df[columns2].to_numpy()
        temp1=abs(temp)

        sortedtmp=temp1[:, temp1[0, :].argsort()]

        med_array=df['d%sdt_med_val' %direc].to_numpy()
        mad_array=df['d%sdt_mad_val' %direc].to_numpy()

        kh_array=np.median(sortedtmp,axis=1)/(med_array+3*mad_array)

        temp=np.column_stack((sortedtmp,med_array,mad_array,kh_array))
        columns=['%s max1'%direc,'%s max2'%direc,'%s max3'%direc,'%s max4'%direc,'%s med'%direc, '%s mad'%direc, '%s snr'%direc]
        ##
        #temp=df[columns].to_numpy()
        
        # power spectra are in a np array within the dataframe
        powers=pd.DataFrame(df['d%sdt_power' %direc].tolist(), index= df.index).to_numpy()[:,1:30]
        names=columns+['%s_P%s' %(direc,f) for f in np.arange(powers.shape[1])]
        
        # concatenate features with power spectra
        X_temp=np.hstack((temp,powers))
        X_list+=[X_temp]
        name_list_tmp+=[names]
        
        #########TARGET VECTOR#########
        y_temp=df['d%sdt_Y' %direc]
        #Make 3's "maybe" param (1 or 2)
        #y_temp.loc[df['d%sdt_Y' %direc] == 3] = fs_params['maybes']
        
        '''
        # Drop 3's from target and idx from features
        y_temp=y_temp.loc[df['d%sdt_Y' %direc]<3]
        X_temp=X_temp[df['d%sdt_Y' %direc]<3]
        '''
        
        #Change labels to match "positivity" for ML
        # make 1's into 0's and 2's into 1's-- motion is "positive" class
        y_temp.loc[df['d%sdt_Y' %direc] == 1] = 0
        y_temp.loc[df['d%sdt_Y' %direc] == 2] = 1
        
        
        y_list+=[y_temp]
        
    X=np.hstack(X_list)
    name_list=np.hstack(name_list_tmp)
    y_tmp=np.vstack(y_list)
    
    #logic : if any component has motion, say motion is present
    y=np.sum(y_tmp, axis=0)
    y=np.where(y>0, 1, y)
    return X, y, name_list, df.time



def fs_to_Xy_vertical(df, param):
    '''
    function to vertically concatenate velocity components (East, North Up) into feature sets.
    Feature vector m*3 samples x n (features).
    Target vector is a single 3*m x 1, with logic to check each component's label into single label.
    
    "Maybes" are values I labeled as maybe motion.  Maybe variable maps these into 1 (no motion) or 2 (motion)
    '''
    X_list=[]
    y_list=[]
    name_list_tmp=[]
    time_list=[]
    
    #df=df[(df.dedt_Y!=3) & (df.dndt_Y!=3) & (df.dudt_Y!=3)]
    #vertically stack components into single arrays (X, Y)
    
    for direc in param['dims'][:1]: # only include HORIZONTALS FOR NOW
        
        columns=['d%sdt_max_val' %direc, 'd%sdt_max2_val' %direc, 'd%sdt_med_val' %direc, 
                 'd%sdt_min_val' %direc, 'd%sdt_min2_val' %direc, 'd%sdt_mad_val' %direc]
        
        tmp_df=df[df['d%sdt_Y' %direc]!=3] #drop components with 3
        ##
        columns2=['d%sdt_max_val' %direc, 'd%sdt_max2_val' %direc,  'd%sdt_min_val' %direc, 'd%sdt_min2_val' %direc]

        temp=tmp_df[columns2].to_numpy()
        temp1=abs(temp)

        sortedtmp=temp1[:, temp1[0, :].argsort()]

        med_array=tmp_df['d%sdt_med_val' %direc].to_numpy()
        mad_array=tmp_df['d%sdt_mad_val' %direc].to_numpy()

        kh_array=np.median(sortedtmp,axis=1)/(med_array+3*mad_array)

        temp=np.column_stack((sortedtmp,med_array,mad_array,kh_array))
        columns=['%s max1'%direc,'%s max2'%direc,'%s max3'%direc,'%s max4'%direc,'%s med'%direc, '%s mad'%direc, '%s snr'%direc]
        ##
        #temp=df[columns].to_numpy()
        
        
        #columns=['d%sdt_med_val' %direc] #
        
        #temp=tmp_df[columns].to_numpy()
        powers=pd.DataFrame(tmp_df['d%sdt_power' %direc].tolist(), index= tmp_df.index).to_numpy()[:,1:30]
        names=columns+['%s_P%s' %(direc,f) for f in np.arange(powers.shape[1])]
        name_list_tmp+=[names]
        
        X_temp=np.hstack((temp,powers))
        #X_temp=temp #
        X_list+=[X_temp]

        y_temp=tmp_df['d%sdt_Y' %direc]
        #Make 3's 1's
        #y_temp.loc[df['d%sdt_Y' %direc] == 3] = maybes
        
        #Change labels to match "positivity"
        # make 1's into 0's and 2's into 1's-- motion is "positive" class
        y_temp.loc[tmp_df['d%sdt_Y' %direc] == 1] = 0
        y_temp.loc[tmp_df['d%sdt_Y' %direc] == 2] = 1
        y_list+=[y_temp]
        time_list+=[tmp_df.time]

    X=np.concatenate(X_list)
    y=np.concatenate(y_list)
    name_list=np.hstack(name_list_tmp)
    
    return X, y,name_list, time_list

def read_meta(path):
    custom_meta_key = 'feature_meta'
    table = pq.read_table(path)
    meta_json = table.schema.metadata[custom_meta_key.encode()]
    meta = json.loads(meta_json)
    return meta


# ML Functions

def list_to_featurearrays(fold, param):
            '''
            convert list of station/events to concatenated dataframes
            then convert dataframes to feature sets using feature extraction function
            '''
            #fold=pd.read_parquet([os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/',f) for f in fold])
            
            path=[os.path.join(os.path.dirname(os.getcwd()), 'data/feature_set/', '%s_*' %f) for f in fold]
            fold_set_list=[]
            for j in path:
                fold_set_list+=glob.glob(j)
            fold=pd.read_parquet(fold_set_list)
            #X_, y_, name_list, times=fs_to_Xy_horizontal(fold, params)
            if param['stacking'] == 'horizontal':
                X_, y_, name_list, times=fs_to_Xy_horizontal(fold, param)
            if param['stacking'] == 'vertical':
                X_, y_, name_list, times=fs_to_Xy_vertical(fold, param)
            
            
            return X_, y_, name_list, times
        
        
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
        
        X_train, y_train, name_list, times=list_to_featurearrays(train_fold, param)  
        X_test, y_test, name_list, times=list_to_featurearrays(test_fold, param)
        #print((X_train.shape,X_test.shape))
        
        # train classifier
        clf = RandomForestClassifier(n_estimators=param['n_estimators'], max_depth=param['max_depth'], class_weight=param['class_wt'],random_state=10, n_jobs=-1).fit(X_train, y_train)
        
        ###WAS JUST THIS 
        #y_pred=clf.predict(X_test)
        
        y_pred_prob=clf.predict_proba(X_test)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        # locate the index of the largest f score
        ix = argmax(fscore)
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

def grid_search(train_set, params):
    '''
    '''
    result=[]
    for param in params: 
        precision, recall, f1, threshold = k_fold_results(train_set, param)
        result.append((list(param.values()))+list([precision,recall,f1,threshold]))
    df = pd.DataFrame(result, columns=list(param.keys())+['precision','recall','f1','threshold'])
    print(df.sort_values(by=['f1'], ascending=False)[['max_depth', 'n_estimators','class_wt','threshold','f1']])
    i=df.f1.idxmax()
    return(params[i],df.iloc[i])


### EQ UTILS###

def read_vel(array):
    '''
    array is array from li_df dataframe
    '''
    
    #fname='output/%s/velocities_%s_%03d_%s.txt' %(event,station,doy,year)
    fname='output/%s/velocities_%s_%s_%s.txt' %(array.eventID,array.station,array.doy,array.year)
    fpath=os.path.join(os.path.dirname(os.getcwd()), fname)
    vel_df=pd.read_csv(fpath, header=None, sep=' ', usecols=[1,2,3,4,5], names=['gpst','dndt','dedt','dudt','clk'])
    utc_diff=gps_2_utc(vel_df.gpst[0])
    vel_df['utc'] = vel_df.gpst + utc_diff
    vel_df.index=(pd.to_datetime(vel_df.utc, unit='s', origin='unix'))
    return vel_df

def read_vel_fname(fname):
    '''
    
    '''
    
    #fname='output/%s/velocities_%s_%03d_%s.txt' %(event,station,doy,year)
    #fname='output/%s/velocities_%s_%s_%s.txt' %(array.eventID,array.station,array.doy,array.year)
    #fpath=os.path.join(os.path.dirname(os.getcwd()), fname)
    vel_df=pd.read_csv(fname, header=None, sep=' ', usecols=[1,2,3,4,5], names=['gpst','dndt','dedt','dudt','clk'])
    utc_diff=gps_2_utc(vel_df.gpst[0])
    vel_df['utc'] = vel_df.gpst + utc_diff
    vel_df.index=(pd.to_datetime(vel_df.utc, unit='s', origin='unix'))
    return vel_df

def gps_2_utc(gpstime):
    '''
    calculates offset between gps time and utc (unix) time
    '''
    # Convert GPS time to UTC and index
    gps = datetime.datetime(1980,1,6)
    utc = datetime.datetime(1970,1,1)
    diff=gps-utc
    #https://en.racelogic.support/VBOX_Automotive/01General_Information/Knowledge_Base/What_are_GPS_Leap_Seconds%3F
    leap_sec=gpsleapsec(gpstime)
    utc_diff=diff.seconds + diff.days * 86400 - leap_sec #total seconds including leap seconds
    return utc_diff
    #CREATE FUNCTION FOR LEAP SECONDS

def gpsleapsec(gpssec):
    '''
    number of leapseconds at given gps time epoch
    taken from B.Crowell's Snivel package
    '''
    leaptimes = np.array([46828800, 78364801, 109900802, 173059203, 252028804, 315187205, 346723206, 393984007, 425520008, 457056009, 504489610, 551750411, 599184012, 820108813, 914803214, 1025136015, 1119744016, 1167264017])
    leapseconds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    a1 = np.where(gpssec > leaptimes)[0]
    leapsec = len(a1)
    return(leapsec)

from scipy import stats
from scipy.stats import ncx2

def kh_threshold(ds):
    thresh=ds['gv'].median()+3*ds['gv'].mad()
    return thresh
    
def pgv_threshold(ds, quantile_perc):
    
    nc = np.mean(ds['dedt'])**2+np.mean(ds['dndt'])**2+np.mean(ds['dudt'])**2
    df=3
  
    df, nc, loc, scale=ncx2.fit(ds['gv2'].values, fdf=df, fnc=nc)
    #low, up=ncx2.interval(quantile_perc, df, nc, loc=loc, scale=scale)
    up=ncx2.ppf(quantile_perc, df, nc, scale=scale, loc=loc)
    thresh=np.sqrt(up)

    return thresh

def get_features(dataframe, obs):

    f, p=signal.periodogram(x=dataframe.values, fs=5, nfft=5*30)
    #print(1/(f[-1]))
    #print(1/(f[-2]))
    
    feature_ds = pd.DataFrame({'time':[dataframe.index[-1]], obs+'_max_val':[dataframe.nlargest(2)[0]], 
                               obs+'_max2_val':[dataframe.nlargest(2)[1]], obs+'_med_val':[dataframe.median()],
                   obs+'_min_val':[dataframe.nsmallest(2)[0]], obs+'_min2_val':[dataframe.nsmallest(2)[1]],
                               obs+'_mad_val':[dataframe.mad()], obs+'_power':[p], 
                               obs+'_raw':[dataframe.values], obs+'_Y':[1]})
    
    return feature_ds

def distance(
        ep,
        depth,
        gps):
    """
    distance calculates the hypocentral distances between a station and the epicenter

    Parameters
    ----------
    ep is the lat, long of the event
    depth is the depth of the event (km)
    gps is the lat, long of the station

    :return:
       hypocentral distance in km
    """
    dist =  geopy.distance.geodesic( ep,gps ).km
    hypo = np.sqrt(np.power(dist, 2) + np.power(depth, 2) )
    return hypo

def calculate_p_s_arrival(station, eq):
    
    """
    Function calculates arrival times for P and S waves
    
    Parameters
    ----------
    :param station: gtsm_station class object
        Must include the following attributes
        :station.lat: float
        :station.long: float
    :param eq: earthquake class object
        Must include the following attributes
        :eq.lat: float
        :eq.long: float
        :eq.time: datetime.datetime
    :return:
        Adds attributes to station object
        :station.p_delta: datetime.timedelta 
        :station.s_delta: datetime.timedelta
        :station.p_arrival: datetime.datetime
        :station.s_arrival: datetime.datetime
    """
    
    event_loc="["+str(eq.lat)+","+str(eq.long)+"]"
    station_loc="["+str(station.lat)+","+str(station.long)+"]"
    url = "https://service.iris.edu/irisws/traveltime/1/query?evloc="+event_loc+"&staloc="+station_loc
    df=pd.read_table(url, sep="\s+", header=1, index_col=2, usecols=[2,3])
    
    station.p_delta = datetime.timedelta(seconds=float(df.iloc[(df.index == 'P').argmax()].Travel))
    station.s_delta = datetime.timedelta(seconds=float(df.iloc[(df.index == 'S').argmax()].Travel))
    station.p_arrival = eq.time + station.p_delta
    station.s_arrival = eq.time + station.s_delta
    return station

def calculate_p_s_arrival_depth(station, eq):
    
    """
    Function calculates arrival times for P and S waves
    
    Parameters
    ----------
    :param station: gtsm_station class object
        Must include the following attributes
        :station.lat: float
        :station.long: float
    :param eq: earthquake class object
        Must include the following attributes
        :eq.lat: float
        :eq.long: float
        :eq.time: datetime.datetime
    :return:
        Adds attributes to station object
        :station.p_delta: datetime.timedelta 
        :station.s_delta: datetime.timedelta
        :station.p_arrival: datetime.datetime
        :station.s_arrival: datetime.datetime
    """
    
    event_loc="["+str(eq.lat)+","+str(eq.long)+"]"
    station_loc="["+str(station.lat)+","+str(station.long)+"]"
    eq_depth=str(eq.depth)
    url = "https://service.iris.edu/irisws/traveltime/1/query?evloc="+event_loc+"&staloc="+station_loc+"&evdepth="+eq_depth
    df=pd.read_table(url, sep="\s+", header=1, index_col=2, usecols=[2,3])
    
    station.p_delta = datetime.timedelta(seconds=float(df.iloc[((df.index == 'P') | (df.index == 'p')).argmax()].Travel))
    station.s_delta = datetime.timedelta(seconds=float(df.iloc[((df.index == 'S') | (df.index == 's')).argmax()].Travel))
    station.p_arrival = eq.time + station.p_delta
    station.s_arrival = eq.time + station.s_delta
    return station

class gnss_station:
    """
    Class object containing data and metadata for a gnss station
    """
    
    def __init__(self, fourchar):
        self.name = fourchar

    
class earthquake:
    """
    Class object containing parameters for a specific earthquake event
    """
    
    def __init__(self, eventID):
        self.eventID = eventID
        
def load_event_data(eventID):
    """
    Function loads earthquake parameters for a given event into an earthquake object
       
    Parameters
    ----------
    :param eventID: str
    :return: eq: earthquake class object   
    """
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&eventid="+eventID
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())

    #define earthquake class object
    eq = earthquake(eventID)
    eq.eventID
    eq.name = data["properties"]['title']
    eq.mag = data["properties"]['mag']
    eq.unix_time = data['properties']['time']
    eq.time = datetime.datetime.fromtimestamp(eq.unix_time/1000.0) 
    eq.time= pd.to_datetime(eq.unix_time, unit='ms')
    eq.lat = data["geometry"]['coordinates'][1]
    eq.long = data["geometry"]['coordinates'][0]
    eq.depth = data["geometry"]['coordinates'][2]
    
    #write_event_coords(eq)
    return eq

def rt_class(row,threshold, fs_params,clf, y_,times, save_plot=False):
    vel_df=read_vel(row)
    eq=load_event_data(row.eventID)
    
    #get site coords, use iris webservice to estimate p/s wave arrival
    site=row.station
    station = gnss_station(site)
    station.lat=sta_coords_df[sta_coords_df.stations==site].latitude.values[0]
    station.long=sta_coords_df[sta_coords_df.stations==site].longitude.values[0]
    station=calculate_p_s_arrival_depth(station, eq)
    
    hypo=np.sqrt(np.power(row.radius_from_event, 2) + np.power(eq.depth, 2) )
    # generate plots
    feature_list=[]

    window_size=30

    obs_list=['dedt', 'dndt', 'dudt']

    #calculate number of overlapping windows
    num_win=(len(vel_df)-len(vel_df)%(window_size*5))/((window_size/3)*5)

    win_start=vel_df.index[0]
    z=0
    while win_start < (vel_df.index[-1] - datetime.timedelta(seconds=window_size)):
        win_stop=win_start+datetime.timedelta(seconds=window_size)
        temp_df=vel_df.loc[win_start:win_stop]

        tmp_list=[]
        for i,obs in enumerate(obs_list):
            
            #features
            features =get_features(temp_df[obs], obs)
            tmp_list.append(features)
        feature_list.append(tmp_list[0].merge(tmp_list[1], on='time').merge(tmp_list[2], on='time'))

        win_start+=datetime.timedelta(seconds=1)
        z+=1
    feature_df=pd.concat(feature_list)
    #add integer indices
    feature_df=feature_df.reset_index(drop=True)


    X_rt, y_rt, names,times_rt=fs_to_Xy_horizontal(feature_df, fs_params)

    y_rt_pred=clf.predict(X_rt)


    y_rt_pred_prob=clf.predict_proba(X_rt)[:, 1]
    y_rt_pred = (y_rt_pred_prob >= threshold).astype('int')

    firstd=times_rt[y_rt_pred!=0]
    #firstd=firstd[firstd>row.p_arrival]
    firstd=firstd[firstd>station.p_arrival]
    first_labeled=times[y_!=0].min()-datetime.timedelta(seconds=30)
    firstd=firstd[firstd>first_labeled]
    firstd=firstd.min()
    
    #(times_rt[y_rt_pred>0]-eq.time).dt.total_seconds()
    
    
    if save_plot:
        plot_rt(vel_df, site, eq, station, times_rt, y_rt_pred, y_,times, hypo, firstd)
        
        first_s_rel=(firstd-eq.time).total_seconds()-(station.s_arrival-eq.time).total_seconds()
        if (first_s_rel<0):
            print(row.eventID,site,eq.mag,hypo, first_s_rel)
    
    return firstd, hypo, station


def plot_rt(vel_df, site, eq, station, times_rt, y_rt_pred, y_,times, hypo, firstd):
    
    ######estimate thresholds
    stop_time=pd.to_datetime(eq.time)
    start_time=(stop_time-datetime.timedelta(minutes=2))

    mask = (vel_df.index > start_time) & (vel_df.index <= stop_time)
    noise_df=vel_df.loc[mask]

    #estimate a bunch of threshold values 
    # append to metadata?  
    # probably store in table instead
    noise_df['gv2']=noise_df.dndt**2+noise_df.dedt**2+noise_df.dudt**2
    noise_df['gv']=np.sqrt(noise_df['gv2'])

    kh_thresh=kh_threshold(noise_df)
    ncx2_995_thresh=pgv_threshold(noise_df, .995)
    ncx2_999_thresh=pgv_threshold(noise_df, .999)
    
    ##############
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(5, 1, figsize=(8,8), sharex=True, facecolor='white')

    #%matplotlib inline
    # Set the style
    #plt.style.use('fivethirtyeight')
    max_x=(times_rt[y_rt_pred>0]-eq.time).dt.total_seconds().max()+20
    max_x=30
    min_x=-5
    
    #max_y=vel_df[min_x:max_x].dudt.max()
    mask = (vel_df.index > stop_time) & (vel_df.index <= times_rt[y_rt_pred>0].max())
    max_y=abs(vel_df.loc[mask].dedt).max()
    
    
    axes[0].plot((vel_df.index-eq.time).total_seconds(), vel_df.dedt, label='5hz', color='#377eb8')
    #axes[0].plot((vel_df.dndt.resample('2S').median().index-eq.time).total_seconds(), vel_df.dedt.resample('2S').mean(), color='tab:orange',label='2s running mean')
    axes[0].set_ylim([-max_y, max_y])
    axes[0].legend()
    axes[0].set_title('%s : Mag:%.1f at %skm : east velocity' %(site,eq.mag,int(hypo)))
    #axes[0].axvline(x=

    axes[1].plot((vel_df.index-eq.time).total_seconds(), vel_df.dndt, color='#377eb8')
    #axes[1].plot((vel_df.dndt.resample('2S').median().index-eq.time).total_seconds(), vel_df.dndt.resample('2S').mean(), color='tab:orange')
    axes[1].set_ylim([-max_y, max_y])
    axes[1].set_title('north velocity')

    axes[2].plot((vel_df.index-eq.time).total_seconds(), vel_df.dudt, color='#377eb8')
    #axes[2].plot((vel_df.dndt.resample('2S').median().index-eq.time).total_seconds(), vel_df.dudt.resample('2S').mean(), color='tab:orange')
    axes[2].set_ylim([-max_y, max_y])
    axes[2].set_title('vertical velocity')

    vel_df['gv']=np.sqrt(vel_df.dedt**2+vel_df.dndt**2+vel_df.dudt**2)
    [axes[3].axvline(x=i, color='tab:orange', linewidth=5, alpha=.5) for i in (vel_df[vel_df.gv>kh_thresh].index-eq.time).total_seconds()]


    axes[3].plot((vel_df.index-eq.time).total_seconds(), vel_df['gv'], color='#4daf4a')
    axes[3].axhline(ncx2_995_thresh, color='tab:red', label='NCX2 threshold')
    axes[3].set_title('PGV')
    axes[3].legend()
    axes[3].set_ylim([0, 3*max_y])

    #axes.axhline(y=i for i in )

    # draw gridlines
    #axes[4].grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    #axes[4].set_xticks(np.arange(-.5, 10, 1));


    #[axes[4].axvspan(xmin=i, xmax=i+(30), ymin=0, ymax=1, alpha=.5) for i in (sta.time[y_pred>0]-eq.time).dt.total_seconds()]
    #.axvspan(xmin=i, xmax=i+(30), ymin=0, ymax=1, alpha=.5) for i in (sta.time[y_>0]-eq.time).dt.total_seconds()]
    axes[4].set_yticks(np.arange(0, 3, 1))

    ##[axes[4].add_patch(Rectangle((i-30, 0), 30, 1,color="#e41a1c", alpha=.1)) for i in (times[y_pred>0]-eq.time).dt.total_seconds()]

    [axes[4].add_patch(Rectangle((i-30, 1), 30, 1,color='#984ea3', alpha=.2)) for i in (times[y_>0]-eq.time).dt.total_seconds()]
    #[axes[4].add_patch(Rectangle((i-1, 1), 1, 1,color='gray', alpha=.75)) for i in (times[y_>0]-eq.time).dt.total_seconds()]

    axes[4].set_title('Random Forest 1sps Classifier')
    #axes[1].plot((vel_df.index-eq.time).total_seconds(), vel_df.dndt)
    #axes[2].plot((vel_df.index-eq.time).total_seconds(), vel_df.dudt)
    #axes[0].set_xticks((sta.time-eq.time).dt.total_seconds())
    #
    axes[4].set_xlabel('OT (s)')

    states=['predicted','labeled']
    #ax.set_ylabel('States', fontsize=13)
    #axes[4].set_yticks(np.arange(0, 3, 1))
    axes[4].set_yticks(np.arange(.5, 2.5, 1))
    axes[4].set_yticklabels(states)
    #[axes[3].axvline(x=i, color='tab:orange', linewidth=5, alpha=.5) for i in (vel_df[vel_df.gv>ncx2_995_thresh].index-eq.time).total_seconds()]

    #[axes[4].axvline(x=i, ymin=0, ymax=1,color="#e41a1c") for i in (times_rt[y_rt_pred>0]-eq.time).dt.total_seconds()]
    [axes[4].axvline(x=i, ymin=0, ymax=1,color="#e41a1c") for i in (times_rt[(y_rt_pred>0)&(times_rt>firstd)]-eq.time).dt.total_seconds()]
    #axes[4].set_yticklabels([])
    
    axes[0].set_xlim([-5,max_x])
    
    for b in np.arange(5):
        axes[b].axvline((station.p_arrival-eq.time).total_seconds(), color='black', linestyle='--')
        axes[b].axvline((station.s_arrival-eq.time).total_seconds(), color='black', linestyle=':')
    
    
    fig.tight_layout()

    plt.savefig('rt_plots/%s_%s_RT.png' %(site,eq.eventID), facecolor=fig.get_facecolor(),)
    plt.clf()
    
import requests
from io import StringIO

url='https://web-services.unavco.org/gps/metadata/sites/v1?minlatitude=-180&maxlatitude=180&minlongitude=-180&maxlongitude=180&starttime=&endtime=&summary=false'
req = requests.get(url)
data = StringIO(req.text)

sta_coords_df = pd.read_csv(data)

sta_coords_df=sta_coords_df.drop_duplicates(subset=sta_coords_df.columns[0])[[sta_coords_df.columns[0],sta_coords_df.columns[2],sta_coords_df.columns[3]]]
sta_coords_df.rename(columns={sta_coords_df.columns[0]:'stations'}, inplace=True)
sta_coords_df["stations"]=sta_coords_df.stations.str.lower()

from scipy import stats
from scipy.stats import ncx2