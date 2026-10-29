#----------------------------------------------------------
# Calculate vertical advection for mixed layer heat budget
# Shay Magahey, 03/26/2024, UCSB
# ML heat budget equation based on Graham et al 2014
# Calculates upwelling, thermocline, and vertical eddy terms (terms 5-7 in the Graham eqn)
# 7/18 changed terms (thermo, upwelling, vert eddy, etc) to negative since negative in the overall heat budget equation
#-----------------------------------------------------------
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy.ma as ma
import datetime
import cftime

def vertical_advection_rd_ml(MLD,lat,lon,int_T,int_U,int_V,sub_int_W,sub_int_T):
    #INputs:
    # MLD: make sure right size
    # lat, lon
    # int_U, int_V: make sure right size
    # sub_int_W
    # int_T, sub_int_T
    # 
    #---------------------------------
    # 1. Compute entrainment velocity
    #----------------------------------
    
    RE=6378000
    latarr=lat*np.pi/180
    lonarr=lon*np.pi/180

    #---------------------------------
    # 2. Time, horizontal derivatives of MLD
    #----------------------------------

    # SAM: implementing fixes to time, x/y derivatives as in advection_rd_ml.py
    #dx=MLD.lon.diff('lon')
    dH=MLD.diff('time')    # difference in nanoseconds, convert to seconds
    dt=MLD.time.diff('time')*1e-9
    dHdx=MLD.diff('lon')/(MLD.lon.diff('lon')*(np.pi/180)*np.cos(latarr))/RE# dH here should be difference in x-dimension not time
    #dHdx=dHdx*np.pi/180) # convert units to C/m from C/degree
    
    dHdy=MLD.diff('lat')/(MLD.lat.diff('lat')*(np.pi/180))/RE
    dHdt=dH/dt.astype('float')




    #---------------------------------
    # 3. Entrainment velocity
    #----------------------------------

    dHdX=(dHdx.convert_calendar("proleptic_gregorian",use_cftime=False))
    print(dHdX)
    dHdY=(dHdy.convert_calendar("proleptic_gregorian",use_cftime=False))
    
    w_entr= dHdt + int_U*dHdX+ int_V*dHdY + sub_int_W
    print("w_entr",w_entr.shape)
    # no months in w_entr
    # do you need to divide changeH_change_time by 86400 like is done in the matlab code?

    #---------------------------------
    # 4. Apply reynolds decomp to entrainment velocity,cross-MLD temperature difference
    #----------------------------------

    #MLD=MLD.convert_calendar("proleptic_gregorian",use_cftime=False)
    dTdz=(int_T-sub_int_T)/MLD; # Tmld, Tsub inputs


    #---------------------------------
    # 5. Compute climatological means
    #----------------------------------

    # Climatological mean vertical temperature gradient
    Tm_mld=int_T.groupby('time.month').mean()
    
    Tm_sub=sub_int_T.groupby('time.month').mean()
    # these are the problem for the size mismatch I think
    print("Tm_sub.shape",Tm_sub.shape)
    # tm sub and tm mld both 15,12,10,50
    
    # climatological mean entrainmenrt velocity
    wm=w_entr.groupby('time.month').mean()
 # climatological mean temperature difference
    dTm=Tm_mld-Tm_sub
    print("dTm shape=",dTm.shape)
    

    #---------------------------------
    # 6. Heaviside function
    #----------------------------------

    wsgn=wm
    wsgn = wsgn.where(wsgn > 0, 0) # wsgn > 0 is true, wsgn < 0 is false, replace false w 0
    wsgn = wsgn.where(wsgn == 0, 1) # wsgn=0 is true, not zero is false, replace false with 1
    #wsgn= wsgn.where(wsgn !=0, np.nan) # wsgn != is true, so 0 is false, replace false with nan
    print("wsgn shape=",wsgn.shape)
                     
    #wsgn[wsgn==0]=np.nan

    #---------------------------------
    # 7. Calculate various advection terms
    #----------------------------------

    mn_w_dTdz=(wsgn*wm*dTm)/MLD
    wtmp= w_entr.groupby('time.month')-wm # value seems fine
    wtmp['month'] = wtmp['time'].dt.month
    print("wtmp shape=",wtmp.shape)
    # anomaly of vertical velocity = wtmp
    print(wtmp.time)

    
    upwelling=(-1*wsgn*wtmp.groupby('time.month')*dTm)/MLD.isel(time=slice(1,360)).groupby('time.month') # .isel(time=slice(1,360))upwelling # wtmp is anomaly? of w? (multiplied by -1 bc subtracted in hb eqn)
    
    # upwelling: anomaly w times mean MLD temp minus mean sub-MLD temp divided by MLD
    tmp1=int_T.groupby('time.month')-Tm_mld # T' MLD # no month
    print("tmp1 shape",tmp1.shape)
    tmp2=sub_int_T.groupby('time.month')-Tm_sub


    tmp=tmp1-tmp2
    tmp['month']=tmp['time'].dt.month # assigning months within time coordinates
    print("tmp2.shape",tmp2.shape)
    # integrated temp anomalies in ML minus temp anomalies below mixed layer: tmp1-tmp2
    tmp=tmp[:,0:360,:,:] # ensuring consistent time coord lengths


    thnum=-1*wsgn*wm*tmp.groupby('time.month') # multiply by -1 bc subtracted in hb eqn

    thermo=thnum/MLD
    
    ed=wsgn*wtmp.groupby('time.month')
    ednum=-1*ed*tmp # tmp is T' difference, ed is w' (multiply by -1 bc subtracted in hb eqn)

    eddy = ednum/MLD
    updTpbar=eddy.groupby('time.month').mean()
    thermo=xr.where(thermo !=0,thermo, np.nan) # replaces 0 wsgn values with np nan
    eddy=xr.where(eddy !=0, eddy, np.nan) # replaces 0 wsgn values w np nan
    upwelling=xr.where(upwelling !=0, upwelling, np.nan) # replaces 0 wsgn values w npnan. maybe not necessary but just in case

    dTpdz=tmp/MLD
    dTmdz=dTm/MLD.groupby('time.month').mean()

    # returns a bunch of intermediates and final products
    # wsgn: heaviside
    # wm: mean w
    # w_entr: entrainment velocity
    # wtmp: w anomaly
    # tmp: T' mld - T' sub
    # upwelling, thermo, eddy
    # dTm: Tmmld - Tmsub
    # tmp1, tmp2: T' MLD, T' sub

    
    return dHdt,wsgn,wm,w_entr,wtmp,tmp,upwelling,thermo,eddy,updTpbar,dTm,dHdX,dHdY,Tm_mld,Tm_sub,tmp1,tmp2,dTpdz,dTmdz






