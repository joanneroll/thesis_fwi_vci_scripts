##########################################
#dataprep_analysis.py

#resample all relevant datasets to target resolution of sentinel 3
#temporal extract of the time series for june, july, august, september
#yearly extract also possible
#clipping bounding box to andalucia region 
#author: johanna roll
#2022
##########################################

import xarray as xr
import geopandas as gpd
import rioxarray
from shapely.geometry import mapping 
import os
import dask

import aux_functions

########### set parameters
area_str = "andalucia"
crs = 4326

months_fireseason = [6,7,8,9]
chunks_x = 50
chunks_y = 50 

#define  paths to data and output folder
#era5 land indices
path_era5_land_indices = r"D:\roll_thesis\data_fwi_vi\output\masked_indices_data\fwi_indices_era5land_masked_andalucia_2015-05-02_2021-12-31.nc"
#era5 indices
path_era5_indices = r"D:\roll_thesis\data_fwi_vi\output\masked_indices_data\fwi_indices_era5_masked_andalucia_2016-01-01_2021-12-31.nc"
#ndvi data
path_ndvi = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\mosaik_30_days"
path_ndvi_smoothed = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\ndvi_smoothed"
path_ndvi_difference = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\ndvi_difference"

#burned area data
path_ba = r"D:\roll_thesis\data_fwi_vi\output\ba_preproc"
#vci data
path_vci = r"D:\roll_thesis\data_fwi_vi\output\vci_preproc" 

#shapefile
area_path = r"D:\roll_thesis\data_fwi_vi\input\shape\andalucia_general_ed.shp"

#s3 dummy for resampling to s3 grid
s3_dummy_path = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\s3_dummy\s3_dummy.nc"

#output folder
path_output = "D:\\roll_thesis\\data_fwi_vi\\output\\analysis_preproc"
if not os.path.exists(path_output):
    os.makedirs(path_output)

###########################

def clip_fireseason(ds, months = [6,7,8,9]):
    #select months June, July, August, September from dataset
    ds = ds.sel(time=ds.time.dt.month.isin(months))

    return ds


def prep_data_timeseries_yearly(data,data_str, path_output, area, s3_dummy, timerange=["2016", "2017", "2018", "2019", "2020","2021"]):
    '''
    preparing timeseries:
    - selecting single year
    - resampling to s3 dummy (if necessary)
    - clipping bounding box to area of interest

    input parameter:
    - timeseries (xarray)
    - name of data for identification/naming of output (str)
    - path to general output folder (str) (folder for prepared timeseries will be created within this function)
    - area (shapefile) for clipping timeseries extent
    - s3_dummy (xarray) : template for desired resolution/georeference (here: sentinel 3 data)
    - timerange (list of str): default years 2016-2021

    returns: 
    None
    -> saves prepared timeseries to files in respective folder  
    
    '''
    print(f"prepping fireseason (JJAS) per year: {data_str}")
    for year in timerange:
        print(f"year {year}")
        data_year = data.sel(time=year)
        data_year = data_year.persist()
        #select fire season months
        data_year = clip_fireseason(data_year)

        #clip region to more detailed shape of aoi
        data_year = data_year.transpose("time","y","x",...)
        data_year = aux_functions.check_fix_raster_georeference(data_year,s3_dummy)
        data_year = data_year.rio.clip(area.geometry.apply(mapping),area.crs, drop=True) #from_disk=True)

        folder_out = f"{path_output}\\{data_str}"
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        path_out = f"{folder_out}\\{data_str}_{year}.nc"
        #write data
        #encoding issue -> dynamical encoding parameter
        #other maybe cleaner solution: del data_year.data_vars.attrs["grid_mapping"]
        #dataarray to dataset if not dataset
        try:
            data_year = data_year.to_dataset()
        except:
            data_year = data_year

        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in data_year.data_vars}
        data_year.to_netcdf(path_out, encoding=encoding)  

        del data_year

def prep_data_timeseries_monthly(data,data_str, path_output, area, s3_dummy, months = [6,7,8,9]):
    '''
    preparing timeseries:
    - selecting months of available years
    - resampling to s3 dummy (if necessary)
    - clipping bounding box to area of interest
    
    input parameter:
    - timeseries (xarray)
    - name of data for identification/naming of output (str)
    - path to general output folder (str) (folder for prepared timeseries will be created within this function)
    - area (shapefile) for clipping timeseries extent
    - s3_dummy (xarray) : template for desired resolution/georeference (here: sentinel 3 data)
    - timerange (list of str): default months Jun - Sept

    returns: 
    None
    -> saves prepared timeseries to files in respective folder 
    '''
    print(f"prepping fireseason (JJAS) per month: {data_str}")
    for month in months:
        print(f"month {month}")
        data_month = data.sel(time = data.time.dt.month.isin(month))
        data_month = data_month.persist()   

        #clip region to more detailed shape of aoi
        data_month = data_month.rio.clip(area.geometry.apply(mapping),area.crs, drop=True) #from_disk=True)
        # aux_functions.print_raster(data_month)
        #reproject data to template if necessary
        data_month = data_month.transpose("time","y","x",...)
        data_month = aux_functions.check_fix_raster_georeference(data_month,s3_dummy)
        # print("after potential resampling")
        # aux_functions.print_raster(data_month)


        folder_out = f"{path_output}\\{data_str}"
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)
        
        path_out = f"{folder_out}\\{data_str}_month{month}.nc"
        
        #write data
        #encoding issue -> dynamical encoding parameter
        #dataarray to dataset if not dataset
        try:
            data_month = data_month.to_dataset()
        except:
            data_month = data_month

        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in data_month.data_vars}
        data_month.to_netcdf(path_out, encoding=encoding)  

        del data_month

###########################
#silence warning 
dask.config.set({"array.slicing.split_large_chunks": False})

#read shape
area = gpd.read_file(area_path,crs=f"epsg:{crs}")
#load dummy for s3 dummy
s3_dummy = xr.load_dataarray(s3_dummy_path).rio.write_crs(crs)
#clip dummy to area
s3_dummy = s3_dummy.rio.clip(area.geometry.apply(mapping), area.crs, drop = True)

print("start prepping data")
#########################################
#preparing fwi data per year
#keep fwi & dc // drop ffmc,dmc,isi,bui
 
#### ERA5LAND
data = xr.load_dataset(path_era5_land_indices, drop_variables = ["ffmc", "dmc", "isi","bui"], 
        chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#clip fireseason 
data = clip_fireseason(data)

prep_data_timeseries_monthly(data.fwi, "fwi_era5land", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data.fwi, "fwi_era5land", path_output, area, s3_dummy)

# ERA5
data = xr.load_dataset(path_era5_indices, drop_variables = ["ffmc", "dmc", "isi","bui"], 
        chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#era5 data with y coordinate offset: bounds: (351.625, 35.375, 359.125, 39.625) --> transform
data = data.assign_coords(x=(data.x-360))
#clip fireseason 
data = clip_fireseason(data)

prep_data_timeseries_monthly(data.fwi, "fwi_era5", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data.fwi, "fwi_era5", path_output, area, s3_dummy)

######################
# BURNED AREA
data = xr.open_mfdataset(f"{path_ba}\\*.nc").rio.write_crs(crs) #, parallel = True, chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#clip fireseason 
data = clip_fireseason(data)
prep_data_timeseries_monthly(data, "burned_area", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data, "burned_area", path_output, area, s3_dummy)

# NDVI
#30 days bookkeeping
data = xr.open_mfdataset(f"{path_ndvi}\\*.nc").rio.write_crs(crs) #parallel = True, chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#clip fireseason 
data = clip_fireseason(data)
prep_data_timeseries_monthly(data.ndvi, "ndvi", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data.ndvi, "ndvi", path_output, area, s3_dummy)

#weakly smooth of 30 days bookkeping ndvi
data = xr.open_mfdataset(f"{path_ndvi_smoothed}\\*.nc").rio.write_crs(crs) #parallel = True, chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#clip fireseason 
data = clip_fireseason(data)
prep_data_timeseries_monthly(data.ndvi, "ndvi_smoothed", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data.ndvi, "ndvi_smoothed", path_output, area, s3_dummy)

# NDVI difference   
data = xr.open_mfdataset(f"{path_ndvi_difference}\\*.nc").rio.write_crs(crs) #parallel = True, chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#clip fireseason 
data = clip_fireseason(data)
prep_data_timeseries_monthly(data.ndvi, "ndvi_difference", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data.ndvi, "ndvi_difference", path_output, area, s3_dummy)

# VCI
data = xr.open_mfdataset(f"{path_vci}\\*.nc").rio.write_crs(crs) #, parallel = True, chunks={"y":chunks_y, "x":chunks_x, "time":-1}).rio.write_crs(crs)
#clip fireseason 
data = clip_fireseason(data)
prep_data_timeseries_monthly(data.vci, "vci", path_output, area, s3_dummy)
# prep_data_timeseries_yearly(data.vci, "vci", path_output, area, s3_dummy)


print("prepping data for analysis done.")
