# consolidating all my old versions of composite_functions to this one

# Shay Magahey UCSB 2025

import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time
import numpy.ma as ma

# first selects EN/LN years. The second then uses that output to composite ENSO events around the peak
# you can use any variable in the second function since the first function uses SST to get years



def select_events(tos, n34, regbox, strt, fin, timespi, threshold=1.0):
    """
    Identify El Niño and La Niña years for each ensemble member.

    Parameters
    ----------
    tos      : SST DataArray, detrended but not anomaly
    n34      : Niño3.4 box [lat_min, lat_max, lon_min, lon_max]
    regbox   : broader region box (same format)
    strt     : start year (int)
    fin      : end year (int)
    timespi  : ['YYYY-MM', 'YYYY-MM'] time slice strings
    threshold: SD multiplier for event detection (default 1.5)

    Returns
    -------
    ninoyrs_mem, ninayrs_mem : (member, year) — event years or NaN
    """
    # Isolate regions and time period
    ens_n34 = tos.sel(lat=slice(n34[0],    n34[1]),    lon=slice(n34[2],    n34[3]),    time=slice(timespi[0], timespi[1]))
    ens_sst = tos.sel(lat=slice(regbox[0], regbox[1]), lon=slice(regbox[2], regbox[3]), time=slice(timespi[0], timespi[1]))

    # Anomalies
    ens_n34 = ens_n34.groupby('time.month') - ens_n34.groupby('time.month').mean()
    ens_sst = ens_sst.groupby('time.month') - ens_sst.groupby('time.month').mean()

    # DJF quarterly means
    ens_n34_djf = ens_n34.resample(time='QS-DEC').mean('time')
    ens_n34_djf = ens_n34_djf.where(ens_n34_djf.time.dt.month == 12, drop=True)
    ens_sst_djf = ens_sst.resample(time='QS-DEC').mean('time')
    ens_sst_djf = ens_sst_djf.where(ens_sst_djf.time.dt.month == 12, drop=True)

    # Event masks using threshold
    n34_std = ens_n34_djf.std('time').mean('member').squeeze()
    msk_en  = ens_n34_djf >  threshold * n34_std
    msk_ln  = ens_n34_djf < -threshold * n34_std

    # Build per-member year arrays
    yrrange = np.arange(strt, fin, 1)

    def build_yr_array(mask):
        yrs_per = []
        for i in range(len(mask.member)):
            A = mask.isel(member=i).mean('lat').mean('lon')
            yrs_per.append(xr.where(A[0:len(yrrange)], yrrange, np.nan))
        arr = np.stack(yrs_per, axis=0)
        return xr.DataArray(arr).rename({'dim_0': 'member', 'dim_1': 'time'})

    return build_yr_array(msk_en), build_yr_array(msk_ln)

def var_composites(data,yrs,ninoyrs_mem,ninayrs_mem,nmonths):
    #----------------------------------------------
    # data = variable of interest to composite. SST, wind, DIC, salinity, whatever
    # yrs: integer array of the years that correspond to the time of data
    # ninoyrs_mem: ninoyrs as returned by select_events
    # ninayrs_mem: ninayrs as returned by select_events
    # nmonths: number of months to calculate composite over. Ideally 24 or 48
    # returns EN composite, LN composite with member, time, lat, lon
    # ---------------------------------------------------
    memEN_fb=[]

    # get anomaly of input data. data should be detrended but not anomaly
    tosa=data.groupby('time.month') - data.groupby('time.month').mean()
    # loop through members, ID nino yrs for this member, drop nan years
    for m in range(tosa.shape[0]):
        thismem=xr.DataArray(ninoyrs_mem[m,:]) #.rename({'dim_0':'time'}) # set of yrs for this member
        thisyrsen=yrs.where(~thismem.isnull(),drop=True) # drop nans, not isnull() keeps where there are actual values

        varEN=tosa[m,:]
        varEN=varEN.convert_calendar("proleptic_gregorian",use_cftime=False,align_on='date') # convert time
        monthsEN=[]

        for s in range(1,len(thisyrsen)): 
            precmonths=nmonths/2 # months before en peak
            start=np.array(thisyrsen[s].values,dtype='datetime64[M]')-np.timedelta64(int(precmonths),'M')
            end=start+np.timedelta64(nmonths,'M')
            var1=varEN.sel(time=slice(start,end)).isel(time=slice(0, nmonths)) 
            monthsEN.append(var1) # append the 24 month periods
            
        # this enumerate loop is the key to getting xr.concat to stack correctly
        for i,var in enumerate(monthsEN):
            var['time'] = np.arange(nmonths)  # Assign a consistent time coordinate
            monthsEN[i] = var

        # concatenate each event for this member
        events_fb_en=xr.concat(monthsEN, dim='event')
        # average events per member to get averate strength/evolution
        meaneventsEN_fb=events_fb_en.mean('event')
        # append event average per member
        memEN_fb.append(meaneventsEN_fb) # append for each ensemble member

    # add along ensemble members, event average evolution per member
    monthsmemsEN_fb=xr.concat(memEN_fb,dim='member',coords='minimal')


    
    # does the same as the above loop but for la nina
    memLN_fb=[]
    for m in range(tosa.shape[0]):

        thismem=xr.DataArray(ninayrs_mem[m,:]) #.rename({'dim_0':'time'}) # set of yrs for this member
        thisyrsln=yrs.where(~thismem.isnull(),drop=True) # drop nans, not isnull() keeps where there are actual values
        varLN=tosa.isel(member=slice(0,len(ninayrs_mem.member))) # var of interest, this member
        varLN=varLN.isel(member=m)
        varLN=varLN.convert_calendar("proleptic_gregorian",use_cftime=False,align_on='date') # convert time
        monthsLN=[]
        for s in range(len(thisyrsln)):
            precmonths=nmonths/2
            start=np.array(thisyrsln[s].values,dtype='datetime64[M]')-np.timedelta64(int(precmonths),'M')
            end=start+np.timedelta64(nmonths,'M')
            var0=varLN.sel(time=slice(start,end)).isel(time=slice(0, nmonths)) # select 24 month period around peak to isolate
            monthsLN.append(var0)
                
        for i,var in enumerate(monthsLN):
            if len(var['time']) < nmonths:
                # Create an array of length 48 filled with NaNs
                new_time = xr.DataArray(np.arange(nmonths), dims=['time'])
                new_data = xr.DataArray(np.full(nmonths, np.nan), dims=['time'], coords={'time': new_time})

                # Reindex var to match the new time dimension and fill with NaNs where necessary
                var = var.reindex(time=new_time, fill_value=np.nan)
                monthsLN[i] = var
            else:
                # Ensure the time coordinate has a consistent length of 48
                var['time'] = np.arange(nmonths)  # Assign a consistent time coordinate
                monthsLN[i] = var


        events_fb_ln=xr.concat(monthsLN, dim='event')
        meaneventsLN_fb=events_fb_ln.mean('event')
        memLN_fb.append(meaneventsLN_fb) # append for each ensemble member

    monthsmemsLN_fb=xr.concat(memLN_fb,dim='member',coords='minimal')
    monthsmemsLN_fb=monthsmemsLN_fb.assign_coords({'time':np.arange(int(-1*nmonths/2),int(nmonths/2),1)})
    monthsmemsEN_fb=monthsmemsEN_fb.assign_coords({'time':np.arange(int(-1*nmonths/2),int(nmonths/2),1)})
    #monthsmemsEN_sst=xr.concat(memEN_sst,dim='member')

    return monthsmemsEN_fb,monthsmemsLN_fb

# also: function for basic DJF map composites

def get_masks(tos,regbox,n34,times):
    #tos=tos-tos.mean('member')
    ens_sst=tos.sel(lat=slice(regbox[0],regbox[1]),lon=slice(regbox[2],regbox[3]),time=slice(times[0],times[1]))
    ens_n34=tos.sel(lat=slice(n34[0],n34[1]),lon=slice(n34[2],n34[3]),time=slice(times[0],times[1]))
        
    # Calculate NINO3.4 regional average
    ens_n34=ens_n34.mean(dim="lat").mean(dim="lon")
                
    # Detrend NINO3.4 by removing the ensemble mean
    #ens_n34=ens_n34-ens_n34.mean('member')
        
    # Remove climatology
    clim=ens_n34.groupby('time.month').mean()
    ens_n34=ens_n34.groupby('time.month')-clim  
    clim=ens_sst.groupby('time.month').mean()
    ens_sst=ens_sst.groupby('time.month')-clim  

    # Calculate DJF averages
    tmpdjf=ens_n34.resample(time='QS-DEC').mean(dim="time")
    #print(tmpdjf)
    tmpdjf=tmpdjf.where(tmpdjf.time.dt.month == 12,drop="True")
    #print(tmpdjf)
    ens_n34_djf=tmpdjf
    #print(ens_n34_djf)
    
    tmpdjf=ens_sst.resample(time='QS-DEC').mean(dim="time") # quarterly averages
    tmpdjf=tmpdjf.where(tmpdjf.time.dt.month == 12,drop="True") # Isolates DJF quarters only
    ens_sst_djf=tmpdjf    
    
    # Identify El Nino, La Nina years
    tmpn34std=ens_n34_djf.std('time').mean('member') # one value: overall std dev
    #print('std=',tmpn34std)
    msk_en=ens_n34_djf > tmpn34std # Isolates years that are el nino years 
    en_evt=ens_sst_djf.where(msk_en, drop=True).mean('member')
    # average SST anomaly across times and members for years that ARE el nino years

    msk_ln=ens_n34_djf < -1*tmpn34std
    ln_evt=ens_sst_djf.where(msk_ln, drop=True).mean('member')

    return en_evt,ln_evt,ens_n34

# functions for strong events/my events
def select_strong_events(tos,n34,regbox,strt,fin,timespi,thr):
    ens_n34=tos.sel(lat=slice(n34[0],n34[1]),lon=slice(n34[2],n34[3]),time=slice(timespi[0],timespi[1]))

    ens_sst=tos.sel(lat=slice(regbox[0],regbox[1]),lon=slice(regbox[2],regbox[3]),time=slice(timespi[0],timespi[1]))
    #ens_n34=tos.sel(lat=slice(ninobox[0],ninobox[1]),lon=slice(ninobox[2],ninobox[3]),time=slice(times[0],times[1]))
    
    clim=ens_n34.groupby('time.month').mean()
    ens_n34=ens_n34.groupby('time.month')-clim  

    clim=ens_sst.groupby('time.month').mean()
    ens_sst=ens_sst.groupby('time.month')-clim

    tmpdjf=ens_n34.resample(time='QS-DEC').mean(dim="time")
    tmpdjf=tmpdjf.where(tmpdjf.time.dt.month == 12,drop="True") # n34 sst
    ens_n34_djf=tmpdjf

    tmpdjf1=ens_sst.resample(time='QS-DEC').mean(dim="time")
    tmpdjf1=tmpdjf1.where(tmpdjf1.time.dt.month == 12,drop="True") # all sst
    ens_sst_djf=tmpdjf

    tmpn34std=ens_n34_djf.std('time').mean('member')
    msk_en=ens_n34_djf > thr*tmpn34std
    en_evt=ens_sst_djf.where(msk_en, drop=True)

    msk_ln=ens_n34_djf < -1*thr*tmpn34std
    ln_evt=ens_sst_djf.where(msk_ln, drop=True)

    yrrange=np.arange(strt,fin,1)

    enyrs_per=[]
    # Filter out nans/fill values

    for i in range(len(en_evt.member)):
        A=msk_en.isel(member=i).mean('lat').mean('lon')
        fut=xr.where(A[0:30],yrrange,np.nan)
        enyrs_per.append(fut)
    ninoyrs_mem=np.stack(enyrs_per,axis=0)
    lnyrs_per=[]
    for i in range(len(ln_evt.member)):
        B=msk_ln.isel(member=i).mean('lat').mean('lon')
        futln=xr.where(B[0:30],yrrange,np.nan)
        lnyrs_per.append(futln) 
    ninayrs_mem=np.stack(lnyrs_per,axis=0)

    return(ninoyrs_mem,ninayrs_mem)


# get my/sy
def get_sy_my_LN(enyrs,lnyrs,fb,yr1,yr2,yrs,nmonths):
    # enyrs: list of enyrs
    # lnyrs: list of lnyrs
    # fb= varibale/feedback of interest (i.e. temp tendency), sst- anomaly should already be calculated if var of interest
    # nmonths: total number of months in period around peak: ie +/- 2 years is 48 months
    mems=np.array([0,2,3,4,5,6,7,8,10,11,12,13])
    memsMY_LN=[]
    memsSY_LN=[]
    yrs=yrs.rename({'dim_0': 'time'})
    for m in range(len(fb.member)-1):
        thismem = xr.DataArray(lnyrs[m, :],dims=['time']).assign_coords({'time':np.arange(yr1,yr2,1)})  # Set of years for this member
        #print('thismem=',thismem)
        thisyrsln = yrs.where(~thismem.isnull(), drop=True)  # Drop NaNs
        #print('thisyrsln=',thisyrsln)
        thismemEN = xr.DataArray(enyrs[m, :],dims=['time']) # Set of years for this member
        thisyrsen = yrs.where(~thismemEN.isnull(), drop=True)
        thisyrsen = thisyrsen #.values.astype(str).astype(int)
        varLN = fb[m, :]
        varLN = varLN.convert_calendar("proleptic_gregorian", use_cftime=False)
        
        months_myLN = [] # initialize for appending
        months_syLN = []
        for s in range(len(thisyrsen)):
            start = np.array(thisyrsen[s], dtype='datetime64[M]') - np.timedelta64(int(nmonths/2), 'M')
            end = start + np.timedelta64(nmonths, 'M')
            var0 = varLN.sel(time=slice(start, end)).isel(time=slice(0, nmonths))
            if var0.values.size<nmonths*len(fb.lat)*len(fb.lon):
                shape = (nmonths, len(fb.lat),len(fb.lon))
                varnLN = xr.DataArray(np.full(shape, np.nan),dims=('time','lat','lon'))
            else:
                varnLN=var0
            cond1= int(thisyrsen[s])+1
            cond2=int(thisyrsen[s])+2
            thisyrsenn = thisyrsen.astype(str).astype(int)
            thisyrslnn = thisyrsln.astype(str).astype(int)
            if cond1 in thisyrslnn and cond2 in thisyrslnn: # two consecutive LNs after EN?
                
                shape=(nmonths, len(fb.lat),len(fb.lon))
                varMY_LN=varnLN
                #print('multiyear la nina')
                varSY_LN=np.full(shape,np.nan)
            elif cond1 in thisyrslnn: #
                #print('single year la nina')
                # preceding year is la nina year
                shape=(nmonths, len(fb.lat),len(fb.lon))
                varSY_LN=varnLN
                varMY_LN=np.full(shape,np.nan)

            else:
                shape=(nmonths, len(fb.lat),len(fb.lon))
                varMY_LN=np.full(shape,np.nan)
                varSY_LN=np.full(shape,np.nan)
            
            if varMY_LN.size < nmonths*len(fb.lat)*len(fb.lon):
                shape=(nmonths,len(fb.lat),len(fb.lon))
                varMY_LN=np.full(shape,np.nan)
            elif varSY_LN.size < nmonths*len(fb.lat)*len(fb.lon):
                shape=(nmonths, len(fb.lat),len(fb.lon))
                varSY_LN=np.full(shape,np.nan)
            print('appending individual events for this member')
            months_myLN.append(xr.DataArray(varMY_LN,dims=['time','lat','lon']).assign_coords({'time':np.arange(-nmonths/2,nmonths/2,1)}))
            # problem with cesm2
            months_syLN.append(xr.DataArray(varSY_LN,dims=['time','lat','lon']).assign_coords({'time':np.arange(-nmonths/2,nmonths/2,1)}))
        myLNmonths=xr.concat(months_myLN,dim='event',coords='minimal',compat='override').mean('event')
        syLNmonths=xr.concat(months_syLN,dim='event',coords='minimal',compat='override').mean('event')
        print('appending all members for this model')
        memsMY_LN.append(myLNmonths)
        memsSY_LN.append(syLNmonths)
    myLN=xr.concat(memsMY_LN,dim='new',coords='minimal',compat='override')
    syLN=xr.concat(memsSY_LN,dim='new',coords='minimal',compat='override')
    print('concatenating all members for this model for final result')
    return myLN,syLN


# EN-LN and LN-LN isolate
def get_types_strongLN(enyrs,lnyrs,fb,yr1,yr2,yrs,nmonths):
    # enyrs: list of enyrs
    # lnyrs: list of lnyrs
    # fb= varibale/feedback of interest (i.e. temp tendency)
    # nmonths= whole period number of months
    mems=np.array([0,2,3,4,5,6,7,8,10,11,12,13])
    mems_wEN=[]
    mems_noEN=[]
    mems_wLN=[]
    yrs=yrs.rename({'dim_0': 'time'})
    for m in range(len(mems)):
        thismem = xr.DataArray(lnyrs[m, :],dims=['time']).assign_coords({'time':np.arange(yr1,yr2,1)})  # Set of years for this member
        thisyrsln = yrs.where(~thismem.isnull(), drop=True)  # Drop NaNs
        thismemEN = xr.DataArray(enyrs[m, :],dims=['time']) # Set of years for this member
        thisyrsen = yrs.where(~thismemEN.isnull(), drop=True)
        thisyrsen = thisyrsen.values.astype(str).astype(int)
        varLN = fb[m, :]
        varLN = varLN.convert_calendar("proleptic_gregorian", use_cftime=False)
        
        
        months_wEN = [] # initialize for appending
        months_noEN = []
        months_wLN = []
        for s in range(len(thisyrsln)):
            #print('LN year=',thisyrsln[s].values)
            start = np.array(thisyrsln[s].values, dtype='datetime64[M]') - np.timedelta64(int(nmonths/2), 'M')
            end = start + np.timedelta64(int(nmonths), 'M')
            var0 = varLN.sel(time=slice(start, end)).isel(time=slice(0, nmonths))
            #print(var0.time)
            #print('var0 size',var0.values.size)
            if var0.values.size<nmonths*len(fb.lat)*len(fb.lon):
                shape = (nmonths, len(fb.lat), len(fb.lat))
                varnLN = xr.DataArray(np.full(shape, np.nan),dims=('time','lat','lon'))
            else:
                varnLN=var0
            #monthsLN.append(xr.DataArray(varnLN).assign_coords({'time':np.arange(-12,12,1)}))
            #print('for mem '+str(m)+' varln=',varnLN)
            cond1= int(thisyrsln[s].values)-1
            
            #print('cond=',cond)
            #print('thisyrsen',thisyrsen)
            thisyrslnn = thisyrsln.values.astype(str).astype(int)
            if cond1 in thisyrsen: # if preceding year is an el nino
                
                shape=(nmonths,len(fb.lat),len(fb.lon))
                varEN = varnLN
                var_noEN=np.full(shape,np.nan)
                LN_precedes=np.full(shape,np.nan)
            elif cond1 in thisyrslnn:
                # preceding year is la nina year
                shape=(nmonths,len(fb.lat),len(fb.lon))
                LN_precedes=varnLN
                varEN=np.full(shape,np.nan)
                var_noEN=np.full(shape,np.nan)
            
            else: # no preceding LN and not a shoulder year
                #print('prec yr is not EN')
                shape=(nmonths,len(fb.lat),len(fb.lon))
                #print('no preceding EN')
                var_noEN=varnLN
                #print('var_noEN size=',var_noEN.shape)
                #print('var_noEN time=',var_noEN.time)
                varEN=np.full(shape,np.nan)
                LN_precedes=np.full(shape,np.nan)
            if varEN.size < nmonths*len(fb.lat)*len(fb.lon):
                shape=(nmonths,len(fb.lat),len(fb.lon))
                varEN=np.full(shape,np.nan)
            elif var_noEN.size < nmonths*len(fb.lat)*len(fb.lon):
                shape=(nmonths,len(fb.lat),len(fb.lon))
                var_noEN=np.full(shape,np.nan)
            elif LN_precedes.size < nmonths*len(fb.lat)*len(fb.lon):
                shape=(nmonths,len(fb.lat),len(fb.lon))
                LN_precedes=np.full(shape,np.nan)
            months_wEN.append(xr.DataArray(varEN,dims=['time','lat','lon']).assign_coords({'time':np.arange(-nmonths/2,nmonths/2,1)}))
            months_noEN.append(xr.DataArray(var_noEN,dims=['time','lat','lon']).assign_coords({'time':np.arange(-nmonths/2,nmonths/2,1)}))
            months_wLN.append(xr.DataArray(LN_precedes,dims=['time','lat','lon']).assign_coords({'time':np.arange(-nmonths/2,nmonths/2,1)}))
        precENmonths=xr.concat(months_wEN,dim='event',coords='minimal').mean('event')
        noprecENmonths=xr.concat(months_noEN,dim='event',coords='minimal').mean('event')
        precLNmonths=xr.concat(months_wLN,dim='event',coords='minimal').mean('event')
        mems_wEN.append(precENmonths)
        mems_noEN.append(noprecENmonths)
        mems_wLN.append(precLNmonths)
    avgd_precEN_months=xr.concat(mems_wEN,dim='member',coords='minimal')
    avgd_noprecEN_months=xr.concat(mems_noEN,dim='member',coords='minimal')
    avgd_precLN_months=xr.concat(mems_wLN,dim='member',coords='minimal')
    return avgd_precEN_months,avgd_precLN_months,avgd_noprecEN_months

def plot_composites(lon,lat,data,regbox,cmap,vmin,vmax,cblabel,cbticks,plttitle):
    """
    Preliminary plots to show mean state changes in future relative to PI and PD
    data= data to plot, 2D (array) # ensst_arr
    regbox: region to plot (array)
    cmap=colormap (str)
    vmin,vmax: range for data (int)
    cblabel: colorbar label (str)
    cbticks: colorbar ticks (array)
    plttitle: plot title (str)
    figname: name.png (str)
    returns figure
    """
    
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree(central_longitude=180))
    proj = ccrs.PlateCarree(central_longitude=180)
    ax.set_extent([regbox[2], regbox[3], regbox[0], regbox[1]], crs=ccrs.PlateCarree())
    im=plt.pcolormesh(lon,lat,data,transform = ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax)
    
    plt.title(plttitle,fontsize=25)
    #bounds1=np.linspace(-3,3,50)
    cbar1 = plt.colorbar(im,label=cblabel, orientation="vertical",extend = 'both',ax=ax,drawedges=False)
    cbar1.set_ticks(cbticks)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
    ax.coastlines()
    #plt.savefig(figname)
    return

def quarter_only(var,qstr,qmonth):
    djf=var.resample(time=qstr).mean(dim="time")
    djf=djf.where(djf.time.dt.month == qmonth,drop="True") # n34 sst
    tot_djf=djf
    return tot_djf

