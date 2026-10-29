# ----------------------------------------------------------
# Calculate horizontal advection for mixed layer heat budget
# Shay Magahey, 03/26/2024, UCSB
# ML heat budget equation based on Graham et al 2014
# Calculates ubar*delT', ekman, and horizontal eddy terms (terms 2-4 in the Graham eqn)

# LOOK AT SST GRAD CALCS
# -----------------------------------------------------------

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy.ma as ma
import datetime
import cftime

def advection_rd_ml(temp,lat,lon,time,Uh,Vh):
    # probably need to isolate Uh, Vh over ML before inputting
    # 1. Calculate horizontal tracer gradient
    RE=6378000
    latarr=lat*np.pi/180
    lonarr=lon*np.pi/180

    dTdx=temp.diff('lon')/(temp.lon.diff('lon')*(np.pi/180)*np.cos(latarr))/RE  
    dTdy=temp.diff("lat")/(temp.lat.diff('lat')*np.pi/180)/RE # 
    # SAM: moved time conversion up to beginning of code
    dTdx=dTdx.convert_calendar("proleptic_gregorian",use_cftime=False)
    dTdy=dTdy.convert_calendar("proleptic_gregorian",use_cftime=False)
    
    # Convert units to C/m by dividing by radius of Earth
    #dTdx=dTdx
    #dTdy=dTdy
    # double counted

    #------------------------------------------------------------
    # 2. Get climatological mean salinity and currents
    #------------------------------------------------------------
    # from advection_ml_rd.m
    # monthly grouped means of currents and salt

    Tm=temp.groupby('time.month').mean()
    Um=Uh.groupby('time.month').mean()
    Vm=Vh.groupby('time.month').mean()
    print("Um=",Um[0,4,5,60])

    # Print values for sanity checking
    # Mean temperature: get value for May
    '''
    print("month")
    print(Tm.month[4])
    print("Tm")
    print(Tm[0,4,5,60])
    '''

    #------------------------------------------------------
    # 3. Gradient of climatological mean temp
    #------------------------------------------------------
    # SAM: modified this since the "differentiate" function already does the division by the denominator
    # so you don't need to do that again
    #dSmdx=Sm.differentiate('lon')*(np.pi/180)*np.cos(latarr)/RE # included the factor of pi/180 since lon/lat are in degrees
    #dSmdy=Sm.differentiate('lat')*np.pi/180./RE

    dTmdx=Tm.diff('lon')/(Tm.lon.diff('lon')*(np.pi/180)*np.cos(latarr))/RE # included the factor of pi/180 since lon/lat are in degrees
    dTmdy=(Tm.diff('lat'))/(Tm.lat.diff('lat')*np.pi/180)/RE

    arrU=Um*dTmdx
    arrV=Vm*dTmdy
    
    mn_adv_clim_u=xr.DataArray(np.tile(arrU, (1, 1, 1)),coords=arrU.coords, dims=arrU.dims) 
    mn_adv_clim_v=xr.DataArray(np.tile(arrV, (1, 1, 1)),coords=arrV.coords, dims=arrV.dims)
 
    #print("mn_adv_clim_u")
    #print(mn_adv_clim_u)
    
    #-------------------------------------------------------
    # 4. Anomalous advection of climatological mean gradient
    #-------------------------------------------------------
    # from advection_ml_rd.m
    UmUp=(Uh.groupby('time.month')-Um) #U anomaly
    VmVp=(Vh.groupby('time.month')-Vm) # need to groupby
    
    anomadv_U=UmUp.groupby('time.month')*dTmdx # 
    anomadv_V=VmVp.groupby('time.month')*dTmdy

    # u' * dTbar/dx, v'*dTbar/dy
    # SAM: changed to return these terms separately, since we'll often want to consider just one
    UpdTmdx = -1*anomadv_U # ekman
    VpdTmdy = -1*anomadv_V  # ekman

    #-------------------------------------------------------
    # 5 Climatological mean advection of anomalous gradient
    #-------------------------------------------------------
    # from advection_ml_rd.m

    climadv_U=Um*(dTdx.groupby('time.month') - dTmdx).groupby('time.month') 
    climadv_V=Vm*(dTdy.groupby('time.month') - dTmdy).groupby('time.month')

    dTpdx=dTdx.groupby('time.month') - dTmdx # which direction will zonal gradient be?
    dTpdy=dTdy.groupby('time.month') - dTmdy
    # SAM: changed names of variables since I kept getting confused by them

    print('dTpdx=',dTpdx[0,100,5,60])
    
    UmdTpdx = -1*climadv_U 
    VmdTpdy = -1*climadv_V

    #-------------------------------------------------------
    # 6 Anomalous advection of anomalous gradient
    #-------------------------------------------------------
    anomadv_anomU=(Uh.groupby('time.month')-Um)*(dTdx.groupby('time.month') - dTmdx)
    anomadv_anomV=(Vh.groupby('time.month')-Vm)*(dTdy.groupby('time.month') - dTmdy)
    # overall value minus mean gives anomaly

    # SAM: also separated zonal/meridional terms here
    eddyU = -1*anomadv_anomU 
    eddyV = -1*anomadv_anomV
    mneddyU=eddyU.mean('time')
    mneddyV=eddyV.mean('time')

    #-------------------------------------------------------
    # 7. Convert times to seconds to make python not angry about time derivative
    #-------------------------------------------------------
    time=time.convert_calendar("gregorian",use_cftime=False)

    #-------------------------------------------------------
    # 8. Time Derivative
    #-------------------------------------------------------

    # SAM: modified to make units correct, I have provided two different methods below
    # ALSO: modified to calculate the ANOMALOUS time tendency instead of the TOTAL time tendency
    Sp = temp.groupby('time.month') - temp.groupby('time.month').mean()
    
    # Method 1: using diff
    '''
    num=Sp.diff('time')
    den=time.diff('time')*1e-9 # difference in nanoseconds, convert to seconds
    dTtmp=num.astype('float')/den.astype('float')
    # SAM: the indexing on the dSpdt array here seems weird, I can't locate the numbers that seem like they should be there
    # I think the isel is not actually shifting the array - replaced with xarray "shift"
    #dSpdt=0.5*(dStmp.isel(time=slice(0,-1))+dStmp.isel(time=slice(1,None)))
    dTpdt=0.5*(dTtmp+dTtmp.shift(time=1))
    '''
    '''
    print("num")
    print(num[0,99:102,5,60])
    print("den")
    print(den[99:102])
    print("dTtmp")
    print(dTtmp[0,99:102,5,60])
    print("dTpdt")
    print(dTpdt[0,99:102,5,60])
    '''
          
    # Method 2: using differentiate (xarray calculates the actual derivative with respect to time, not just the difference)
    dTpdt = Sp.differentiate("time") * 1e9  # units returned by differentiate are C/ns, convert to C/s

    # Return results
    return dTpdx,dTpdy,mn_adv_clim_u, mn_adv_clim_v, UpdTmdx, VpdTmdy, UmdTpdx, VmdTpdy, eddyU, eddyV, mneddyU, mneddyV, dTpdt, UmUp, VmVp, dTmdx, dTmdy,Um,Vm


