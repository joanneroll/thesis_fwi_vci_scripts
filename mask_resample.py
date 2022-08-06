#########################################
#mask_resample.py

#mask era5 land calculated indices
#mask era5 historical indices (reference) and resample to area of interest (era5 land)

#apply era5 land sea mask:
# grid boxes where this parameter has a value above 0.5 can be comprised of a mixture of land and inland water but not ocean. 
# Grid boxes with a value of 0.5 and below can only be comprised of a water surface

#author: johanna roll
#2022
##########################################


from os import listdir
from os.path import join
import xarray as xr

import aux_functions 

#data paths
#folder for clipped era5 data (preprocessed data)
# (AOI slighly larger that area of era5 land calculated indices) 
#data time range 01/2016-12/2021
#area: generourly clipped to area andalucia
data_indices_era5_yearly = r"C:\Users\johan\Desktop\master_data\data\output\clipped_era5"

#land-sea-mask (lsm) of era5 data (no preprocessing)
#data time range 01/2016-12/2012
#area europe
data_lsm_era5 = r"C:\Users\johan\Desktop\master_data\data\era5_land_sea_mask\era5_lsm_europe_2015_2022.grib"

#land-sea-mask of era5 land data
#data time range
data_lsm_era5land = r"C:\Users\johan\Desktop\master_data\data\era5_land_land_sea_mask\lsm_1279l4_0.1x0.1.grb_v4_unpack.nc"

#era5 land calculated indices (preprocessed data)
#data time range 09/2015-12/2021 (earlier start point due to required initializing for drought code calculation)
#area andalucia
data_indices_era5land = r"C:\Users\johan\Desktop\master_data\data\output\indices_calculated\fwi_indices_andalucia_2015-05-02_2021-12-31.nc."

#output folder
data_output = r"C:\Users\johan\Desktop\master_data\data\output\masked_data"

#area
area = "andalucia"

#threshold lsm for masking
mask_threshold = 0.5

#define crs
crs = 4326



###########
#load data 
#assign crs and rename dimensions (for rioxarray to work properly) --> norm_dataarray

#era5-land calculated indices
era5land_indices_orig = xr.load_dataset(data_indices_era5land).rio.write_crs(crs)
era5land_indices_orig = aux_functions.norm_dataarray(era5land_indices_orig)

#era5 lsm
era5_lsm_orig = xr.load_dataset(data_lsm_era5).rio.write_crs(crs)
era5_lsm_orig = aux_functions.norm_dataarray(era5_lsm_orig)

#era5-land lsm
era5land_lsm = xr.load_dataset(data_lsm_era5land).rio.write_crs(crs)
era5land_lsm = aux_functions.norm_dataarray(era5land_lsm)

#load era5 reference indices
#list with available dates in folder
dataarrays_era5_yearly = []
for file in listdir(data_indices_era5_yearly):
    if file.endswith(".nc"):
        file_path = join(data_indices_era5_yearly, file)

        era5_yearly = xr.load_dataset(file_path).rio.write_crs(crs)
        dataarrays_era5_yearly.append(era5_yearly)
#merge along dimension time
era5_indices = xr.merge(dataarrays_era5_yearly)
era5_indices = aux_functions.norm_dataarray(era5_indices)

#############
##masking

#era5 indices (reference)
#check time range of arrays
print("\nsync time range of 'era5_lsm' and 'era5_indices'")
era5_lsm,era5_indices = aux_functions.sync_time_range(era5_lsm_orig,era5_indices)

#reproject/resample era5 lsm 
print("resampling era5_lsm to era5_lsm_res matching era5_indices")
era5_lsm_res = era5_lsm.rio.reproject_match(era5_indices)

#check equal georeference
print("\ncompare resolution of 'era5_lsm_res' and 'era5_indices' following resampling")
aux_functions.check_raster_georeference(era5_lsm_res,era5_indices)

#check exact alignment 
print("check alignment of 'era5_lsm_res' and 'era5_indices'")
era5_lsm_res, era5_indices = aux_functions.check_fix_digits_alignment(era5_lsm_res, era5_indices)


print("masking era5 indices")
#mask era5 indices
era5_indices_masked = era5_indices.where(era5_lsm_res.lsm >= mask_threshold)

##export data
#get date range
date_min = era5_indices_masked.time.min().values
date_max = era5_indices_masked.time.max().values

path_out = f"{data_output}\\fwi_indices_era5_masked_{area}_{str(date_min)[:10]}_{str(date_max)[:10]}.nc"
era5_indices_masked.to_netcdf(path_out)

print("\n----------------------------------------")
print("----------------------------------------")

### mask era5 land indices
## masking era5 land indices with resampled era5 lsm

#era5 land calculated indices
#check time range of arrays
print("\nsync time range of 'era5_lsm' and 'era5land_indices'")
era5_lsm_era5land, era5land_indices_1 = aux_functions.sync_time_range(era5_lsm_orig, era5land_indices_orig)

#reproject/resample era5 lsm 
print("resampling/reprojecting era5_lsm to era5_lsm_era5land_res matching era5land_indices")
era5_lsm_era5land_res = era5_lsm_era5land.rio.reproject_match(era5land_indices_1)

#check equal georeference
print("\ncompare resolution of 'era5_lsm_era5land_res' and 'era5land_indices' following resampling/reprojecting")
aux_functions.check_raster_georeference(era5_lsm_era5land_res,era5land_indices_1)

#check exact alignment
print("check alignment of 'era5_lsm_era5land_res' and 'era5land_indices'")
era5_lsm_era5land_res,era5land_indices_1 = aux_functions.check_fix_digits_alignment(era5_lsm_era5land_res, era5land_indices_1)

print("masking era5 land indices")
#mask era5 lannd indices
era5land_indices_masked = era5land_indices_1.where(era5_lsm_era5land_res.lsm >= mask_threshold)

##export data
#get date range
date_min = era5land_indices_masked.time.min().values
date_max = era5land_indices_masked.time.max().values

path_out = f"{data_output}\\fwi_indices_era5land_masked_{area}_{str(date_min)[:10]}_{str(date_max)[:10]}.nc"
era5land_indices_masked.to_netcdf(path_out)

##################################################################################################################
##################################################################################################################

#resample masked era5 indices (reference) to mask era5 land indices (own calculation)
#resample era5 data to era5 land
print("\nresampling/reprojecting 'era5_indices_masked' to 'era5_indices_masked_res' matching 'era5land_indices_masked'")
era5_indices_masked_res = era5_indices_masked.rio.reproject_match(era5land_indices_masked)

#norm dataarrays
era5_indices_masked_res = aux_functions.norm_dataarray(era5_indices_masked_res)
era5land_indices_masked = aux_functions.norm_dataarray(era5land_indices_masked)

print("compare resolution of 'era5_indices_masked_res' and 'era5land_indices_masked' following resampling/reprojecting")
aux_functions.check_raster_georeference(era5_indices_masked_res,era5land_indices_masked)

#check time range of arrays
print("\ncheck and sync time range of 'era5_indices_masked_res' and 'era5land_indices_masked'")
era5_indices_masked_res, era5land_indices_masked = aux_functions.sync_time_range(era5_indices_masked_res, era5land_indices_masked)

indices_diff = era5land_indices_masked - era5_indices_masked_res

##export data
#get date range
date_min = indices_diff.time.min().values
date_max = indices_diff.time.max().values

path_out = f"{data_output}\\fwi_indices_diff_era5lsm_{area}_{str(date_min)[:10]}_{str(date_max)[:10]}.nc"
indices_diff.to_netcdf(path_out)

print("\n\nexport resampled and masked era5/era5 land data")
print("calculating difference of calculated (era5 land) and reference indices (era5) done.")