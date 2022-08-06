#########################################
#calc_vci.py

#calculation of vegetation condition index
#based on smoothed ndvi (7 days) - ndvi_smooth_difference.py

#author: johanna roll
#2022
##########################################<

import os
import xarray as xr
import numpy as np
from numba import guvectorize
from dask.distributed import Client
import matplotlib.pyplot as plt

#path to directory with preprocessed ndvi
# path_ndvi = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc" 
path_ndvi = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\ndvi_smoothed"
#path for vci output
path_output = r"D:\roll_thesis\data_fwi_vi\output\vci_preproc"
if not os.path.exists(path_output):
    os.makedirs(path_output)

# path_output_smooth = f"{path_output}\\vci_smoothed"
# if not os.path.exists(path_output_smooth):
#     os.makedirs(path_output_smooth)

area = "andalucia"
# smooth = 7

#### 
#script produces warning "RuntimeWarning: invalid value encountered in calc_vci return self.ufunc(*args, **kwargs)"
################################  
@guvectorize(
    "(float32[:], float32[:])",
    "(n) -> (n)",
    nopython=True)

def calc_vci(ds, out):
    #access max and min ndvi value of pixel in timeseries
    ndvi_min = np.nanmin(ds)
    ndvi_max = np.nanmax(ds)
    #iterate over each date of timeseries
    vci_collect = []
    for ndvi_current in ds:  
        #catch error when dividing np.nan
        if ndvi_current:
            vci = (ndvi_current-ndvi_min)/(ndvi_max-ndvi_min) 
        else: 
            vci = ndvi_current #np.nan
        vci_collect.append(vci)

    #return calculated vci timeseries for pixel
    # return np.asarray(vci_collect)
    out[:] = np.asarray(vci_collect)
    
def calc_vci_ufunc_dask(ds):
    return xr.apply_ufunc(
    calc_vci,
    ds,
    input_core_dims= [["time"]],
    output_core_dims=[["time"]],
    dask = "parallelized"
) 

####################
#load ndvi data of timeseries
#open multiple (yearly) files of ndvi with parallel computation (dask)
ndvi = xr.open_mfdataset(f"{path_ndvi}\\*.nc", parallel = True, chunks={"y":50, "x":50, "time":-1}).rio.write_crs(4326)
#calculate vci for ndvi timeseries
print("preparing vci calculation")
#chunk core dimension time into a single dask array chunk
vci = calc_vci_ufunc_dask(ndvi.ndvi.chunk({"time":-1})) 
vci = vci.rename(new_name_or_name_dict = "vci").astype(np.float32)
vci = vci.persist()
print("vci calculated")
#save vci as yearly files 
years = ["2016","2017","2018","2019", "2020","2021"]

print("export vci per year")
for year in years:
    print(f"exporting year {year}")
    path = f"{path_output}\\vci_{area}_{year}.nc"
    vci.sel(time=year).to_netcdf(path)

# #smooth timeseries in e.g. 14 days rolling window
# print(f"smooting vci in {smooth} days rolling window")
# vci = vci.rolling(time=smooth).mean().persist().astype(np.float32)
# print(vci)

# print("export smoothed vci per year (rolling window {smooth} days)")
# for year in years:
#     print(f"exporting year {year}")
#     path = f"{path_output_smooth}\\vci_{area}_{year}_smooth{smooth}.nc"
#     vci.sel(time=year).to_netcdf(path)

print("done.")