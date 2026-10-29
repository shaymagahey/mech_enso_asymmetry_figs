# for the obs data, probably not needed for model
import cftime
import xarray as xr
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import statsmodels.api as sm
from scipy.linalg import svd

def detrend_data(data):
    # Flatten the time and space dimensions for regression
    time = np.arange(data.shape[0])
    space = data.shape[1] * data.shape[2]  # Flatten lat, lon

    # Flatten the data (time, lat, lon)
    data_flat = data.values.reshape((-1, space))

    # Perform linear regression for each spatial point
    trend = np.apply_along_axis(lambda x: stats.linregress(time, x)[0] * time + stats.linregress(time, x)[1], axis=0, arr=data_flat)
    detrended_data = data_flat - trend.reshape((data.shape[0], -1))

    # Reshape back to the original dimensions
    detrended_data = detrended_data.reshape(data.shape)

    return xr.DataArray(detrended_data, coords=data.coords, dims=data.dims)

def regress_out(variable, index):
    """Regress index (CTI) out of a given variable at each grid point."""

    variable = np.asarray(variable)
    index = np.asarray(index)

    #print(f"variable shape: {variable.shape}, index shape: {index.shape}")

    # Ensure both arrays are 1D (over time)
    if variable.ndim != 1 or index.ndim != 1:
        raise ValueError(f"Expected 1D arrays, got {variable.shape} and {index.shape}")

    # Add constant for OLS
    X = sm.add_constant(index)

    # Handle NaNs
    valid = ~np.isnan(variable) & ~np.isnan(index)
    if np.sum(valid) < 2:  # Not enough data for regression
        return np.full_like(variable, np.nan)

    # Perform regression
    model = sm.OLS(variable[valid], X[valid]).fit()
    predicted = model.predict(X)

    # Compute residuals
    residuals = variable - predicted

    #print(f"residuals shape: {residuals.shape}")

    return residuals


def calc_npmm(sst,u,v,ds_sst,times):

    # as of 4.18.25 this has been debugged and validated by the same data as Chiang and Vimont 2004!
    # is SST isolated to NPMM region before input? is U?
    
    sst=sst.sel(time=slice(times[0],times[1]))
    
    sst_detr = sst- sst.mean('member')
    u_detr = u -  u.mean('member')
    v_detr = v - v.mean('member')

    sst_run = sst_detr.rolling(time=3, center=True).mean()
    u_run = u_detr.rolling(time=3, center=True).mean()
    v_run = v_detr.rolling(time=3, center=True).mean()
    
    # Calculate anomalies
    tos_anom=sst_run.groupby('time.month')-sst_run.groupby('time.month').mean()
    U_anom=u_run.groupby('time.month')-u_run.groupby('time.month').mean()
    V_anom=v_run.groupby('time.month')-v_run.groupby('time.month').mean()

    
    # regress out cold tongue index from SST so that NPMM is leading mode of variability
    cti=ds_sst['tos'].sel(lat=slice(-6,6),lon=slice(180,270),time=slice(times[0],times[1])).mean('lat').mean('lon')
    # cti is NOT anomaly while tos_anom is- this is probably causing issues with the values
    cti = cti.groupby('time.month') - cti.groupby('time.month').mean()  # Anomalize
    
    sst_regressed = xr.apply_ufunc(
    regress_out, tos_anom, cti,
    input_core_dims=[["time"], ["time"]],  # Apply along time axis
    output_core_dims=[["time"]],  # Ensure output has the same time dimension
    exclude_dims=set(("time",)),  # Prevent broadcasting over time
    vectorize=True
)


    #sst_regressed = sst_regressed.assign_coords(time=tos['time'])
    #u_regressed = u_regressed.assign_coords(time=u['time'])
    #v_regressed = v_regressed.assign_coords(time=v['time'])

    
    
    # Stack spac dims
    X = sst_regressed.stack(space=("lat", "lon"))
    Y = U_anom.stack(space=("lat", "lon"))
    Z = V_anom.stack(space=("lat", "lon"))
    # condenses lat/lon coordinates into space

    W = xr.concat([Y, Z], dim="space")  # Combined U+V anomalies

    # Normalize values

    X_norm = X / X.std(dim="time")
    W_norm = W / W.std(dim="time")

    # Convert xarray to NumPy for SVD
    X_np = X_norm.transpose("time", "member", "space").values
    W_np = W_norm.transpose("member", "time", "space").values

    # mask nans so you don't get an error
    #xmask = ~np.isnan(X_np) 
    #wmask =  ~np.isnan(W_np)
    #X_masked = xr.where(xmask, X_np, 0)  # Keep ocean points, set land to 0
    #W_masked = xr.where(wmask, W_np, 0)

    X_np = np.nan_to_num(X_np)
    W_np = np.nan_to_num(W_np)

    # Initialize list to store time series
    X_MCA_list = []
    W_MCA_list = []

    # Loop over ensemble members to calculate MCA time series
    st=[]
    for i in range(len(X_norm.member)):  # Loop over members
        X_m = X_np[:, i, :]   # (time, space)
        W_m = W_np[i, :, :]   # (time, space)

        # MCA: SVD of X^T W
        U, s, Vt = svd(X_m.T @ W_m, full_matrices=False)

        # Leading MCA spatial patterns
        X_pat = U[:, 0] # check for sign here?
        W_pat = Vt.T[:, 0]

        # Project to get MCA time series
        x_mca = X_m @ X_pat # multiply matrices to get pattern of variance #/ s[0]  # divide by singular value to normalize
        w_mca = W_m @ W_pat #/ s[0]
        
        # need to check if sign is positive or negative for each member
        
        X_MCA_list.append(xr.DataArray(x_mca, dims="time", coords={"time": X.time}))
        W_MCA_list.append(xr.DataArray(w_mca, dims="time", coords={"time": W.time}))
        st.append(xr.DataArray(s))
    # Stack members
    X_MCA = xr.concat(X_MCA_list, dim="member")
    W_MCA = xr.concat(W_MCA_list, dim="member")
    smode=xr.concat(st,dim='member')
    return X_MCA, W_MCA,smode

    import xarray as xr
    import numpy as np
    from scipy.linalg import svd

def calc_npmm_dask(sst, u, v, ds_sst, times):
    # Slice time
    sst = sst.sel(time=slice(times[0], times[1]))
    u = u.sel(time=slice(times[0], times[1]))
    v = v.sel(time=slice(times[0], times[1]))

    # Detrend / remove ensemble mean
    sst_detr = sst - sst.mean('member')
    u_detr = u - u.mean('member')
    v_detr = v - v.mean('member')

    # Running mean
    sst_run = sst_detr.rolling(time=3, center=True).mean()
    u_run = u_detr.rolling(time=3, center=True).mean()
    v_run = v_detr.rolling(time=3, center=True).mean()

    # Monthly anomalies
    tos_anom = sst_run.groupby("time.month") - sst_run.groupby("time.month").mean()
    U_anom = u_run.groupby("time.month") - u_run.groupby("time.month").mean()
    V_anom = v_run.groupby("time.month") - v_run.groupby("time.month").mean()

    # Regress out CTI
    cti = ds_sst['tos'].sel(lat=slice(-6,6), lon=slice(180,270), time=slice(times[0], times[1])).mean(['lat','lon'])
    cti = cti.groupby("time.month") - cti.groupby("time.month").mean()

    tos_anom = tos_anom.chunk({"time": -1})
    cti = cti.chunk({"time": -1})

    tos_regressed = xr.apply_ufunc(
            regress_out, tos_anom, cti,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized"
        )

    # Stack spatial dims
    X = tos_regressed.stack(space=("lat","lon"))
    U_stack = U_anom.stack(space=("lat","lon"))
    V_stack = V_anom.stack(space=("lat","lon"))
    W = xr.concat([U_stack, V_stack], dim="space_component")

    # Fill NaNs with 0 for SVD
    X_norm = (X / X.std('time')).fillna(0)
    W_norm = (W / W.std('time')).fillna(0)

    # Ensure full chunks along core dims
    X_norm = X_norm.chunk(dict(time=-1, space=-1))
    W_norm = W_norm.chunk(dict(time=-1, space=-1))

    # Align X and W
    X_norm, W_norm = xr.align(X_norm, W_norm, join="inner")

    # SVD function per member
    def member_svd(Xm, Wm):
        U_svd, s, Vt_svd = svd(Xm.T @ Wm, full_matrices=False)
        x_mca = Xm @ U_svd[:,0]
        w_mca = Wm @ Vt_svd.T[:,0]
        return x_mca, w_mca, s[0]  # leading MCA mode only

    # Apply SVD
    result = xr.apply_ufunc(
            member_svd, X_norm, W_norm,
            input_core_dims=[["time","space"], ["time","space"]],
            output_core_dims=[["time"], ["time"], []],  # only leading mode
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float,float,float]
        )

    X_MCA, W_MCA, S = result
    X_MCA = X_MCA.assign_coords(time=X_norm.time)
    W_MCA = W_MCA.assign_coords(time=W_norm.time)

    return X_MCA, W_MCA, S, X_norm

def lead_lag_corr_member(npo, enso, max_lag=12):
    """
    Compute lead-lag correlations between NPO and ENSO for each member using xarray.shift.
    
    Positive lag: NPO leads ENSO
    Negative lag: ENSO leads NPO

    Parameters
    ----------
    npo : xarray.DataArray
        NPO time series (member x time)
    enso : xarray.DataArray
        ENSO time series (member x time)
    max_lag : int
        Maximum lag in time steps (both positive and negative)

    Returns
    -------
    lags : np.ndarray
        Array of lags from -max_lag to +max_lag
    corr_mean : xarray.DataArray
        Mean correlation across members at each lag
    corr_member : xarray.DataArray
        Correlations per member (members x lags)
    """
    lags = np.arange(-max_lag, max_lag + 1,1)
    n_members = npo.sizes['member']
    
    corr_list = []

    # Align the time coordinates
    common_times = np.intersect1d(npo.time.values, enso.time.values)
    npo = npo.sel(time=common_times)
    enso = enso.sel(time=common_times)

    for lag in lags:
        # Shift npo relative to time:
        
        print(lag)
        
        # Compute correlation along 'time' for each member
        corr = xr.corr(npo.shift(time=lag), enso, dim='time')
        # negative: ENSO leads , positive: oscillation leads
        #print(corr) prints correlation across members for each lag
        corr_list.append(corr)

    # Combine into one DataArray along 'lag' dimension
    corr_member = xr.concat(corr_list, dim='lag')
    corr_member = corr_member.assign_coords(lag=lags)
    
    # Compute mean correlation across members
    corr_mean = corr_member.mean(dim='member')

    return lags, corr_mean, corr_member




def calc_npo_index(slp_np,slp_eqp,slp):
    # calc anomalies
    slp_np_anom=slp_np.groupby('time.month')-slp_np.groupby('time.month').mean()
    slp_eqp_anom=slp_eqp.groupby('time.month')-slp_eqp.groupby('time.month').mean()

    slp_anom=slp.groupby('time.month')-slp.groupby('time.month').mean()
    diff=slp_np_anom - slp_eqp_anom

    # want DJF SLPA for compositing
    djf=diff.resample(time='QS-DEC').mean(dim="time")
    djf=djf.where(djf.time.dt.month == 12,drop="True")
    diff_djf=djf
    

    std=diff_djf.std('time').mean('member')
    std = diff.std('time').mean('member')  # scalar across time & members

    # Positive NPO events
    msk_pos = diff > 0.5*std
    npo_pos_events = diff.where(msk_pos)  # keeps member x time

    # Negative NPO events
    msk_neg = diff < -0.5*std
    npo_neg_events = diff.where(msk_neg)
    


    return npo_pos_events,npo_neg_events,diff
