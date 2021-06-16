import xarray as xr
import glob
import os
import datetime
import numpy as np
import pandas as pd

def interp_on_month(argd):
    yeart=argd[:4]
    montht=argd[4:]
    #DATE REF
    date_1 = datetime.datetime.strptime('01/01/1950', "%m/%d/%Y")

    ROOT = '/home1/ballycotton/DATA/PIRATE/COLOC_EN4/'+yeart+'/'
    path = glob.glob(os.path.join(ROOT+'*'+montht+'.nc'))
    print yeart,'-',montht,': ','1-open ',path
    #CUSTOM INDEX ARGO
    ARGO=xr.open_dataset('argo_index_2005-2015_EDW.nc')
    #OPEN
    OCC=xr.open_mfdataset(path,concat_dim='N_OBS',decode_times=False,mask_and_scale=True)
    #MATCH FILTER
    OCC=OCC.where((OCC.MATCH_EN4==1),drop=True)
    #DROP SOME USELESS VARIABLES FOR INTERPOLATION
    OCC=OCC.drop(['JULD','MATCH_EN4','LATITUDE_EN4','LONGITUDE_EN4',
                  'PSAL_EN4','DEPTH_EN4','POTM_EN4','POTM_EN4_QC',
                  'PSAL_EN4_QC','TEMP_EN4'])

    OCC=OCC.rename({'JULD_EN4': 'JULD'})
    OCC=OCC.squeeze()

    #CORRECT Q PREFIX FOR OCCIPUT ARGO WMO
    idq=[i for i,item in enumerate(OCC.STATION_IDENTIFIER.values) if "Q" in item]
    for k in idq:
        aa=str(OCC.STATION_IDENTIFIER[k].values)     
        OCC.STATION_IDENTIFIER.load()
        OCC.STATION_IDENTIFIER[k]=aa[1:]+' '  
    
    print yeart,'-',montht,': ','2a-QC filtering ...'
    l1=len(OCC.N_OBS)
    #QC FILTER
    mask=((~np.isnan(np.abs(OCC.POTM_OBS))) & (~np.isnan(np.abs(OCC.PSAL_OBS))) & \
         (OCC.PSAL_QC == 1.) & (OCC.POTM_QC == 1.) & (~np.isnan(OCC.POTM_Hx)) & (~np.isnan(OCC.PSAL_Hx)))
    OCC=OCC.where((mask.transpose()),drop=True)
    print yeart,'-',montht,': 2b-',l1-len(OCC.N_OBS),' profiles dropped'
    
    # Redundant information through the N_MEMBER dimension:
    vlist = ['PSAL_QC','DEPTH','STATION_IDENTIFIER','PSAL_OBS','JULD',
             'POTM_QC','LONGITUDE','LATITUDE','TEMP_OBS','POTM_OBS']
    for v in vlist:
        OCC[v] = OCC[v].isel(N_MEMBER=0)
    
    # ALIGN VARIABLES ALONG ONE DIMENSION 
    OCC['A']=OCC['JULD'].min(dim='N_LEVELS')
    OCC['C']=OCC['LATITUDE'].min(dim='N_LEVELS')
    OCC['D']=OCC['LONGITUDE'].min(dim='N_LEVELS')
    #reduce string array is more complicated than a min (just on pacnet...)
    #on ballycotton, a simple OCC['B']=OCC['STATION_IDENTIFIER'].min(dim='N_LEVELS') is enough
    aaa=np.sort(OCC.STATION_IDENTIFIER,1)	
    OCC['B']=xr.DataArray(aaa[:,-1],dims='N_OBS')
    #
    OCC=OCC.drop(['JULD','STATION_IDENTIFIER','LATITUDE','LONGITUDE'])
    OCC=OCC.rename({'A':'JULD','B':'STATION_IDENTIFIER','C':'LATITUDE','D':'LONGITUDE'})

    print yeart,'-',montht,': ','3a-Number of data points filtering ...'
    l1=len(OCC.N_OBS)
    #KEEP PROFILES WITH AT LEAST 10 VALUES OVER A 1000M LAYER
    OCC['N']=OCC['POTM_OBS'].notnull().sum('N_LEVELS')
    #OCC['N'] = OCC['N'].isel(N_MEMBER=0)
    H1=OCC['DEPTH'].where(OCC['POTM_OBS'].notnull()).max(dim='N_LEVELS')
    H2=OCC['DEPTH'].where(OCC['POTM_OBS'].notnull()).min(dim='N_LEVELS')
    OCC['H']=H1-H2
    #OCC['H'] = OCC['H'].isel(N_MEMBER=0)
    OCC['KEEP'] = xr.DataArray(np.all((OCC['H'] >= 1000, OCC['N'] >= 10), axis=0),dims= {'N_OBS':OCC['N_OBS']})
    OCC = OCC.where(OCC['KEEP'], drop=True)
    OCC = OCC.drop(['N', 'H', 'KEEP'])
    print yeart,'-',montht,': 3b-',l1-len(OCC.N_OBS),' profiles dropped'
    
    #INTERPOLATION ON STD LEVELS
    from interpClass import InterpProfile
    #STANDARD LEVELS
    sdl = np.arange(0,-1500.,-5.)
    #KEEP PROFILES WITH DEPTH>1500
    print yeart,'-',montht,': ','4a-Profiles < 1500m filtering ...'
    l1=len(OCC.N_OBS)
    OCC=OCC.where(OCC.DEPTH.max('N_LEVELS')>1500,drop=True)
    print yeart,'-',montht,': 4b-',l1-len(OCC.N_OBS),' profiles dropped'

    print yeart,'-',montht,': ','5a-INTERPOLATION ON STD LEVELS'
    #INTERP OBS+MODEL
    interpoler = InterpProfile(axis=sdl, method='linear')
    potm_obs_lin=np.empty([len(OCC.N_OBS),len(sdl)])
    psal_obs_lin=np.empty([len(OCC.N_OBS),len(sdl)])
    temp_obs_lin=np.empty([len(OCC.N_OBS),len(sdl)])
    potm_hx_lin=np.empty([len(OCC.N_MEMBER),len(OCC.N_OBS),len(sdl)])
    psal_hx_lin=np.empty([len(OCC.N_MEMBER),len(OCC.N_OBS),len(sdl)])

    for i in OCC.N_OBS.values:
        potm = OCC['POTM_OBS'].values[i,:] # Profile to interpolate
        psal = OCC['PSAL_OBS'].values[i,:]
        temp = OCC['TEMP_OBS'].values[i,:]
        dpt = -OCC['DEPTH'].values[i,:] 
        potm_obs_lin[i,:] = interpoler.fit_transform(potm, dpt)
        psal_obs_lin[i,:] = interpoler.fit_transform(psal, dpt)
        temp_obs_lin[i,:] = interpoler.fit_transform(temp, dpt)
        for j in OCC.N_MEMBER.values:
            potm_hx = OCC['POTM_Hx'].values[j,i,:] # Profile to interpolate
            psal_hx = OCC['PSAL_Hx'].values[j,i,:]
            potm_hx_lin[j,i,:] = interpoler.fit_transform(potm_hx, dpt)
            psal_hx_lin[j,i,:] = interpoler.fit_transform(psal_hx, dpt)
    print yeart,'-',montht,': ','5b-INTERPOLATION DONE'

    #GET CYCLE NUMBER FROM ARGO INDEX WITH NEAREST DATE
    print yeart,'-',montht,': ','6-GET CYCLE_NUMBER FROM ARGO INDEX'
    cycles=np.array([])
    for k in OCC.N_OBS.values:
        AA=ARGO.where(ARGO.wmo==int(OCC.STATION_IDENTIFIER[k].values),drop=True)
	if len(AA.index)>0:
            dx=date_1 + datetime.timedelta(days=float(OCC.JULD[k].values))
            CC=np.array(dx.strftime('%Y-%m-%dT%H:%M:%S'),dtype='datetime64')
            indw=np.abs(CC-AA.date.values).argmin()
            #print CC,AA.date.values[indw],AA.cycle_cumber.values[indw],np.abs(CC-AA.date.values).min()
            if np.abs(CC-AA.date.values).min() < np.timedelta64(600,'s') :
                cycles=np.append(cycles,int(AA.cycle_cumber.values[indw]))    
            else:
                cycles=np.append(cycles,np.nan)   
	else:
	    cycles=np.append(cycles,np.nan)   

    #BUILD FINAL DATASET
    fds= xr.Dataset({'STATION_IDENTIFIER': (['N_OBS'], OCC.STATION_IDENTIFIER.values),
                     'LATITUDE': (['N_OBS'], OCC.LATITUDE.values), 
                     'LONGITUDE': (['N_OBS'], OCC.LONGITUDE.values), 
                     'JULD': (['N_OBS'], OCC.JULD.values), 
                     'CYCLE_NUMBER': (['N_OBS'], cycles), 
                     'STANDARD_LEVELS': (['N_LEVELS'], sdl),                 
                     'POTM_OBS': (['N_OBS','N_LEVELS'], potm_obs_lin),
                     'POTM_Hx': (['N_MEMBER','N_OBS','N_LEVELS'], potm_hx_lin),
                     'PSAL_OBS': (['N_OBS','N_LEVELS'], psal_obs_lin),                 
                     'PSAL_Hx': (['N_MEMBER','N_OBS','N_LEVELS'], psal_hx_lin),
                     'TEMP_OBS': (['N_OBS','N_LEVELS'], temp_obs_lin)                             
                    })

    fds.JULD.attrs['_FillValue']=99999.0
    fds.JULD.attrs['long_name']='Julian date of observation'
    fds.CYCLE_NUMBER.attrs['_FillValue']=99999.0
    fds.CYCLE_NUMBER.attrs['long_name']='Cycle number retrieved from argo index'
    fds.LATITUDE.attrs['_FillValue']=99999.0
    fds.LATITUDE.attrs['long_name']='Latitude of observation'
    fds.LONGITUDE.attrs['_FillValue']=99999.0
    fds.LONGITUDE.attrs['long_name']='Longitude of observation'
    fds.STANDARD_LEVELS.attrs['_FillValue']=99999.0
    fds.STANDARD_LEVELS.attrs['long_name']='Standard levels'
    fds.POTM_OBS.attrs['_FillValue']=99999.0
    fds.POTM_OBS.attrs['long_name']='Potential temperature of observation interpolated on std levels'
    fds.POTM_Hx.attrs['_FillValue']=99999.0
    fds.POTM_Hx.attrs['long_name']='Model interpolated potential temperature interpolated on std levels'
    fds.PSAL_OBS.attrs['_FillValue']=99999.0
    fds.PSAL_OBS.attrs['long_name']='Practical salinity of observation interpolated on std levels'
    fds.PSAL_Hx.attrs['_FillValue']=99999.0
    fds.PSAL_Hx.attrs['long_name']='Model interpolated practical salinity interpolated on std levels'
    fds.TEMP_OBS.attrs['_FillValue']=99999.0
    fds.TEMP_OBS.attrs['long_name']='Temperature of observation interpolated on std levels'
    #WRITE TO NETCDT
    print yeart,'-',montht,': ','7-Write to netcdf --> ',len(fds.N_OBS),' PROFILES'
    fds.to_netcdf('/home1/ballycotton/DATA/PIRATE/SDL_INTERP/'+yeart+'/OCCITENS-EDW-SDL-'+yeart+montht+'.nc')    
    return

import multiprocessing
#YEARMONTH TO PROCESS
argd0=['201309','201310','201311','201312','201401','201402','201403','201404','201405','201406','201407','201408','201409','201410','201411','201412']

if __name__ == '__main__':
    print "###########"
    print "BEGIN POOL"
    pool = multiprocessing.Pool()
    pool.map(interp_on_month, argd0)
    print "END POOL"

