import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import glob
#from composite_functions import select_events,var_composites
import dask
import statsmodels.api as sm
from scipy.linalg import svd
import sys
sys.path.append("/glade/work/smagahey/UCSB/codes/paper1_new/paper1/python_packages")
from myutils.composite_functions import select_events, var_composites, quarter_only, get_masks
from calc_npmm_npo import regress_out,calc_npmm

timesfut=["2070-01","2099-12"]
timespi=["1850-01","1879-12"]
tipast=["1850-01","2015-01"]
timespd=["1980-01","2009-12"]
tissp=["2015-01","2100-01"]
ep=[-5,5,210,270] #Nino3
n34=[-5,5,190,240] # nino3.4?
n4=[-5,5,160,210]
pac=[-5,5,120,280]
regbox=[-40,40,90,300]
npmmreg=[-21,35,175,265]

ds_sst=xr.open_dataset("/glade/work/smagahey/UCSB/codes/generated_files/tos_1850_2100_E3SMv1LE.nc")
fullsst=ds_sst['tos']
tos=ds_sst['tos'].sel(lat=slice(npmmreg[0],npmmreg[1]),lon=slice(npmmreg[2],npmmreg[3])).sel(time=slice(timespd[0],timespd[1]))
lon=ds_sst['lon']
lat=ds_sst['lat']
dsU=xr.open_dataset("/glade/work/smagahey/UCSB/codes/generated_files/ua_1980_2010_E3SMv1LE.nc")
Upi=dsU['ua'].sel(lat=slice(npmmreg[0],npmmreg[1]),lon=slice(npmmreg[2],npmmreg[3])).isel(member=slice(0,15))
dsV=xr.open_dataset("/glade/work/smagahey/UCSB/codes/generated_files/va_1980_2010_E3SMv1LE.nc")
Vpi=dsV['va'].sel(lat=slice(npmmreg[0],npmmreg[1]),lon=slice(npmmreg[2],npmmreg[3])).isel(member=slice(0,15))

'''
# Flatten MultiIndex into space dimension
tos['lat'] = tos['lat'].astype('float64')
tos['lon'] = tos['lon'].astype('float64')
Upi['lat'] = Upi['lat'].astype('float64')
Upi['lon'] = Upi['lon'].astype('float64')
Vpi['lat'] = Vpi['lat'].astype('float64')
Vpi['lon'] = Vpi['lon'].astype('float64')
'''
X_MCA_pd,W_MCA_pd,s_pd, xnorm_pd = calc_npmm(tos,Upi,Vpi,fullsst,timespd)

X_MCA_pd=xr.Dataset({'X_MCA_pd':X_MCA_pd})

X_MCA_pd.to_netcdf('pd_npmm_e3sm1.nc')




