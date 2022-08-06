#########################################
#ndvi_preprocessing.py

#stack and merge s3 ndvi files
#script written for ndvi files with naming: "andalucia_ndvi_6144_14336_2048_2048_59_10_20160309T102403_wgs84_cog.tif"

#author: johanna roll
#2022
##########################################


import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_datasets
from rioxarray.merge import merge_arrays

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpy as np

from datetime import datetime

import os
from os import listdir
from os.path import join

import aux_functions


#path s3 ndvi folder
s3_dir = r"D:\roll_thesis\data_fwi_vi\input\s3_ndvi"

#path to indices data for setting clip bounds of AOI
path_indices = r"D:\roll_thesis\data_fwi_vi\output\masked_indices_data\fwi_indices_era5land_masked_andalucia_2015-05-02_2021-12-31.nc"

#output folder
path_output_bk = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\mosaik_30_days"
if not os.path.exists(path_output_bk):
    os.makedirs(path_output_bk)

path_output_monthly = f"{path_output_bk}\\monthly_dailymerge_original"
if not os.path.exists(path_output_monthly):
    os.makedirs(path_output_monthly)

path_output_yearly = f"{path_output_bk}\\yearly_dailymerge_original"
if not os.path.exists(path_output_yearly):
    os.makedirs(path_output_yearly)

#get area name dynamically
# area = sorted(listdir(s3_dir))[0].split("_")[1]
area = "andalucia"
crs = "4326"

#load indices data
indices = xr.load_dataset(path_indices).rio.write_crs(4326)
#get bounds and set box for clipping AOI for NDVI data
xmin, ymin, xmax, ymax = indices.rio.bounds()
geodf = gpd.GeoDataFrame(
    geometry=[box(xmin, ymin, xmax, ymax)],
    crs=f"EPSG:{crs}")

#bookkeeping threshold - propagate ndvi values forward up to e.g. 14 days
bk_threshold = 30

###############################
#get first date of available ndvi data
folder_sorted = sorted(listdir(s3_dir),key = lambda x: x.split("_")[8])
# date_proc = sorted(listdir(s3_dir))[0].split("_")[5].split("T")[0]
date_proc = folder_sorted[0].split("_")[8].split("T")[0] #bugfix data
month_proc = date_proc[4:][:2]

#lists for collecting ndvi data of processed day
ndvi_morning = []
ndvi_afternoon = []

#list for merged ndvi data per processed day
ndvi_noon_list = []

#not working RAM wise
#list for collecting monthly stacked ndvi
# ndvi_noon_list_month = []

#counter for progress bar
counter = 0
total_files = len(sorted(listdir(s3_dir)))

print("start iterating over ndvi data")
print("merging ndvi files for each day:")
#iterate over available data
for file in folder_sorted:
    if file.endswith(".tif"):
        #print progress
        aux_functions.progress(counter, total_files)
        counter += 1

        #load data as xarray dataset
        ds = xr.load_dataset(join(s3_dir, file))
        # ds = rxr.open_rasterio(file_path) #load data as dataarray

        #get time info from filename
        # time_str = file.split("_")[5]
        time_str = file.split("_")[8] #bugfix
        #date of aquisition
        date_str = time_str.split("T")[0]
        #month 
        month_str = date_str[4:][:2]
        #time of aquisition
        timeaqu_str = time_str.split("T")[1]
        
        #noon (UTC 12:00:00) threshold for merging ndvis for the morning respectively afternoon of processed day
        if int(timeaqu_str) <= 120000:
            ndvi_morning.append(ds)
        else:
            ndvi_afternoon.append(ds)
                
        #collect ndvi for each date
        #merge ndvi once new date is processed
        if not date_proc == date_str:
            #merge ndvi files before noon
            ndvi_noon = merge_datasets(ndvi_morning, method="first")
            # ndvi_noon = merge_arrays(ndvi_morning) #using rioxarray
            ndvi_noon = ndvi_noon.rio.reproject(f"EPSG:{crs}")
            
            #reorganize data
            ndvi_noon = ndvi_noon.rename({"band_data" : "ndvi"})
            ndvi_noon = ndvi_noon.drop("band").squeeze(dim="band") 
            
            #add time dimension
            dt = datetime.strptime(date_proc, "%Y%m%d").replace(hour=12)
            dt = pd.to_datetime(dt)
            ndvi_noon = ndvi_noon.assign_coords(time = dt)
            ndvi_noon = ndvi_noon.expand_dims(dim='time')
            
            # ndvi_noon = ndvi_noon.expand_dims(time=datetime)
            ndvi_noon_list.append(ndvi_noon)
            
            #shift afternoon ndvi to stack for next day
            ndvi_morning = ndvi_afternoon
            #empty afternoon list
            ndvi_afternoon = []

            #update date tracking variable
            date_proc = date_str            
            # print("\nprocessing date", date_proc)

        #stack data montly (memory usage handling)
        #one month processed, save ndvi to file
        if not month_proc == month_str:
            ndvi_month = xr.combine_by_coords(ndvi_noon_list)
            #clip array to AOI
            ndvi_month = ndvi_month.rio.clip(geodf.geometry, geodf.crs, from_disk=True)
            
            # ndvi_noon_list_month.append(ndvi_month)
            
            #save combined dataset to file
            #retrieve year processed for file name
            year = ndvi_month.time.dt.year.values[0]

            path_out = f"{path_output_monthly}\\ndvi_{area}_{str(year)}_{month_proc}_clipped.nc" 
            ndvi_month.to_netcdf(path_out)  
            del ndvi_month

            # #stack iteratively
            # ndvi_timeperiod = xr.combine_by_coords(ndvi_month)
            # print(ndvi_timeperiod)

            #empty list for next month
            ndvi_noon_list = []

            #update month tracking variable
            month_proc = month_str
            # print("\nprocessing month", month_proc)

#last monthr
ndvi_month = xr.combine_by_coords(ndvi_noon_list)
#clip array to AOI
ndvi_month = ndvi_month.rio.clip(geodf.geometry, geodf.crs, from_disk=True)
year = ndvi_month.time.dt.year.values[0]
path_out = f"{path_output_monthly}\\ndvi_{area}_{str(year)}_{month_proc}_clipped.nc" 
ndvi_month.to_netcdf(path_out)  
del ndvi_month

print("\nmerging and clipping daily ndvi data (per month) finished")

#stacking - not working RAM wise
# ndvi_timeperiod = xr.combine_by_coords(ndvi_month)
# print(ndvi_timeperiod)

# # stacking data for whole time period (not working due to memory error -> exporting stacked data monthly)
# # ndvi_timeperiod = xr.combine_by_coords(ndvi_noon_list)
# ndvi_timeperiod = xr.combine_by_coords(ndvi_noon_list_month)

# date_min = ndvi_timeperiod.time.min().values
# date_max = ndvi_timeperiod.time.max().values

# path_out = f"{path_output}\\ndvi_{area}_{str(date_min)[:10]}_{str(date_max)[:10]}.nc"
# ndvi_timeperiod.to_netcdf(path_out) 


#load monthly saved files and stack per year
print("start stacking daily data (per month) for each year")
print(f"propagate ndvi values up to {bk_threshold} days to fill nan values")

years = ["2016", "2017", "2018", "2019", "2020", "2021"]

arrays_year = []
for year in years:
    print("stacking year: ", year)
    for file in listdir(path_output_monthly):
        if year in file:
            arrays_year.append(xr.load_dataset(join(path_output_monthly, file)))
        
    #stack months per year
    ndvi_year = xr.merge(arrays_year).astype(np.float32)
    ndvi_year.to_netcdf(f"{path_output_yearly}\\ndvi_{area}_{year}_yearly.nc")

    #fill nan values by propagating values forward for max 14 days
    ndvi_year = ndvi_year.ffill("time", limit=bk_threshold).astype(np.float32)
    ndvi_year.to_netcdf(f"{path_output_bk}\\ndvi_{area}_{year}_30.nc")
    
    del ndvi_year
    arrays_year = []
  

print("preprocessing of ndvi indices done.")
print(f"writing stacked and clipped ndvi data (with bookkeeping) to {path_output_bk}.")