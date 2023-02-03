import numpy
import datetime
import calendar
import math
import obspy
from obspy.io.sac import SACTrace
import numpy as np
import os

def gpsleapsec(gpssec):
    leaptimes = numpy.array([46828800, 78364801, 109900802, 173059203, 252028804, 315187205, 346723206, 393984007, 425520008, 457056009, 504489610, 551750411, 599184012, 820108813, 914803214, 1025136015, 1119744016, 1167264017])
    leapseconds = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    a1 = numpy.where(gpssec > leaptimes)[0]
    leapsec = len(a1)
    return(leapsec)
def writesac(velfile,site,stalat,stalon,doy,year,samprate):
    a = numpy.loadtxt(velfile)
    
    tind = a[:,0]
    gtime = a[:,1]
    leapsec = gpsleapsec(gtime[0])
    
    #Get the start time of the file in UTC
    date = datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(doy) - 1)
    gpstime = (numpy.datetime64(date) - numpy.datetime64('1980-01-06T00:00:00'))/ numpy.timedelta64(1, 's')
    stime = (gtime[0]-leapsec)*numpy.timedelta64(1, 's')+ numpy.datetime64('1980-01-06T00:00:00')
    sitem = stime.item()
    styr = sitem.year
    stdy = sitem.day
    stmon = sitem.month
    sthr = sitem.hour
    stmin = sitem.minute
    stsec = sitem.second

    siten='combo'
    nv = a[:,2]
    ev = a[:,3]
    uv = a[:,4]
    
    from pathlib import Path
    path = Path(os.getcwd())
    #sac_dir=str(path.parent.absolute())+'/output/ambient_set/sac/'
    sac_dir='../data/ambient_sac/'
    
    headN = {'kstnm': site, 'kcmpnm': 'LXN', 'stla': float(stalat),'stlo': float(stalon),
             'nzyear': int(year), 'nzjday': int(doy), 'nzhour': int(sthr), 'nzmin': int(stmin),
             'nzsec': int(stsec), 'nzmsec': int(0), 'delta': float(samprate)}
    
    sacn = SACTrace(data=nv, **headN)
    sacn.write(sac_dir+site.upper()  + '_' + doy + '_' + year +  '.vel.n')
    
    headE = {'kstnm': site, 'kcmpnm': 'LXE', 'stla': float(stalat),'stlo': float(stalon),
         'nzyear': int(year), 'nzjday': int(doy), 'nzhour': int(sthr), 'nzmin': int(stmin),
         'nzsec': int(stsec), 'nzmsec': int(0), 'delta': float(samprate)}
    sace = SACTrace(data=ev, **headE)
    sace.write(sac_dir+site.upper()  + '_' + doy + '_' + year +  '.vel.e')

    headZ = {'kstnm': site, 'kcmpnm': 'LXZ', 'stla': float(stalat),'stlo': float(stalon),
             'nzyear': int(year), 'nzjday': int(doy), 'nzhour': int(sthr), 'nzmin': int(stmin),
             'nzsec': int(stsec), 'nzmsec': int(0), 'delta': float(samprate)}
    sacu = SACTrace(data=uv, **headZ)
    sacu.write(sac_dir+site.upper() + '_' + doy + '_' + year + '.vel.u')

    
import pandas as pd
print(os.getcwd())
#pq_file='/home/ec2-user/pgv_ml/synth_gnss_vel/data/ambient_5sps_6yr.pq'
pq_file='../data/jgr_data/ambient_5sps_6yr.pq'
df3=pd.read_parquet(pq_file)

for i in numpy.arange(int(len(df3))):
    site=df3.loc[i].station
    stalat=0
    stalon=0
    doy=df3.loc[i].doy
    year=df3.loc[i].year
    samprate=1/5

    velfile='../data/jgr_data/output/ambient_set_2/velocities_%s_%s_%s.txt' %(site,doy,year)
    try:
        writesac(velfile,site,stalat,stalon,doy,year,samprate)
    except Exception as e:
        print(e)
        pass