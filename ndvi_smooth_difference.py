#########################################
#ndvi_smooth_difference.py

#smoothing ndvi in x days rolling window
#weekly mean

#calculating difference of post ndvi - pre ndvi
#negative values imply decrease in vegetation health
#positive values increase

#author: johanna roll
#2022
##########################################

import xarray as xr
import numpy as np
from numba import guvectorize

import os 
from os import listdir
from os.path import join

import datetime
import pandas as pd

#smoothing data
factor = 7
path = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\mosaik_30_days"
path_out_smooth = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\ndvi_smoothed"
path_out_diff = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\ndvi_difference"

if not os.path.exists(path_out_smooth):
    os.makedirs(path_out_smooth)

if not os.path.exists(path_out_diff):
    os.makedirs(path_out_diff)

## function calculating post - pre difference
@guvectorize(
    "(float32[:], float32[:])",
    "(n) -> (n)",
    nopython=True)

def diff_ndvi_ufunc(ndvi_array, out):
    #guvectorize function for faster computation
    #initialize list: first element is nan
    diff_list = [np.nan]
    
    for i in range(1,len(ndvi_array)):
        #calculate post - prescene
        diff = ndvi_array[i]-ndvi_array[i-1]
        diff_list.append(diff)
     
    #return calculated ndvi diff timeseries for pixel
    # return diff_list
    out[:] = np.asarray(diff_list)
    
def calc_diff_ndvi(ds):
    #calculate ndvi difference
    return xr.apply_ufunc(
        diff_ndvi_ufunc,
        ds,
        input_core_dims= [["time"]],
        output_core_dims=[["time"]])

for file in listdir(path):
    if file.endswith(".nc"):
        filepath = join(path,file)
        print(f"smoothing {file}")
        ds = xr.load_dataset(filepath)
        ds_s = ds.rolling(time=7).mean()

        filename = f"{file.split('.')[0]}_smooth{factor}.nc"
        
        ds_s.to_netcdf(join(path_out_smooth,filename))

        print(f"calc postscene - prescene differenc")
        ds = calc_diff_ndvi(ds.ndvi)
        #analog to burned_area preprocessing
        #day x is information on burn severity on day x
        #for validating wilfire on day x we look at fire danger at day x - 1
        #to simplfy the validation process, we move the date of burn severity at day x 1 day in the past
        time_dt = pd.to_datetime(ds.time.values)
        dates = time_dt.map(lambda t: t - datetime.timedelta(days=1))
        ds["time"] = dates

        filename = f"{file.split('.')[0]}_difference.nc"
        ds.to_netcdf(join(path_out_diff,filename))

