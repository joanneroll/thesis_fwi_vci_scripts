#########################################
#ba_preprocessing_mask.py

#preprocessing of burnt area dataset for burned area mask
#including concatenating of dataframes, merge timeseries for daily steps with threshold 12:00 UTC, rasterize dataframe
#output: binary mask, and mask only for burned area (burned_bin) - not used in final analysis


#author: johanna roll
#2022
##########################################<

import os
from os import listdir
from os.path import join

import geopandas as gpd
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from geocube.api.core import make_geocube
import numpy as np

#path burned area ndvi folder
ba_dir = r"D:\roll_thesis\data_fwi_vi\input\burned_area"

#path to ndvi for setting clip bounds of AOI and resampling the burned area dataset to resolution of analysis
path_ndvi_dummy = r"D:\roll_thesis\data_fwi_vi\output\ndvi_preproc\\s3_dummy\\s3_dummy.nc"

#output folder
path_output = "D:\\roll_thesis\\data_fwi_vi\\output\\ba_preproc"
if not os.path.exists(path_output):
    os.makedirs(path_output)

#get area name dynamically
# area = sorted(listdir(s3_dir))[0].split("_")[1]
area = "andalucia"

#geographic crs
crs = 4326
#projected crs
crs_p = 3035

#load ndvi dummy
s3_dummy = xr.load_dataset(path_ndvi_dummy).rio.write_crs(crs)
#get bounds and set box for clipping AOI for BA data
xmin, ymax, xmax, ymin = s3_dummy.rio.bounds()
geodf = gpd.GeoDataFrame(
    geometry=[box(xmin, ymin, xmax, ymax)],
    crs=f"EPSG:{crs}")

#definition for rasterizing burned area polygons per doy_group
def rasterize_burnedarea_doy (df,raster_like):    
    attributes_list = ["ba_mask", "burned_bin", "burn_severity"] #"area_m", "first_seen_str", "date_index_id", "doy_group", "confidence_total"

    grid = make_geocube(
        vector_data = df,
        measurements = attributes_list,
        like = raster_like
    )
    
    #access timestamp (date / time) from dataframe to set for raster/array for respective doy
    dt = df.iloc[0, :]["date_group"]
    #assign timestamp as dimension of the rasterized xarray data
    grid = grid.assign_coords(time = dt)
    grid = grid.expand_dims(dim = "time")
           
    return grid.astype(np.float32)

################################################
#years being processed
years = ["2016", "2017", "2018", "2019", "2020", "2021"]

##merge gdp
for year in years:
    print("processing: ", year)
    #list collecting all gdf of year
    gpd_year = []

    for file in listdir(ba_dir):
        if year in file:
            path = join(ba_dir, file)
            ba = gpd.read_file(path).to_crs(f'epsg:{crs}')

            ### add meta columns
            #add index as column
            ba["index"] = ba.index+1 #id index with offset of 1 compared to attribute table in qgis
            #clip burned area to aoi
            ba_aoi = gpd.clip(ba,geodf)
            gpd_year.append(ba_aoi)
            # print(ba_aoi.columns)

    #concat dataframes
    ba_aoi_yr_ba = pd.concat(gpd_year, ignore_index=True) #index for identificating polygon saved in column "index"

    #assign 1 for masking
    ba_aoi_yr_ba["ba_mask"] = 1
    #assign "burned area" with class 1
    ba_aoi_yr_ba["burned_bin"] = 1

    ##################################
    #### buffering around burned area ofr "no burned area" with burn_severity=0

    #buffer 300m (laea projection) around burned area
    ba_buffer = ba_aoi_yr_ba.to_crs(crs_p).buffer(200,resolution=15)
    #reproject proj crs back to geog crs
    ba_buffer = ba_buffer.to_crs(crs)
    #geoseries to geodataframe
    ba_buffer = gpd.GeoDataFrame(ba_buffer)
    #add geometry of buffer as column
    ba_buffer = ba_buffer.rename(columns={0:'geometry'}).set_geometry('geometry')
    #isolate "buffer ring" around burned area polygon
    #difference of buffer and polygon
    buffer_ring = ba_buffer.difference(ba_aoi_yr_ba.to_crs(crs), align=True)
    #create geodataframe
    ba_buffer_df = gpd.GeoDataFrame(buffer_ring)
    ba_buffer_df = ba_buffer_df.rename(columns={0:'geometry'}).set_geometry('geometry')
    #add first seen and index information of buffered polygon
    ba_buffer_df["first_seen"] = ba_aoi_yr_ba.loc[ba_aoi_yr_ba.index == ba_buffer_df.index, ["first_seen"]]

    #set confidence to 0 --> buffer to be identified
    # ba_buffer_df["confidence_total"] = no_ba #ba_aoi_yr_ba.loc[ba_aoi_yr_ba.index == ba_buffer_df.index, ["confidence_total"]] 
    #assign 1 for masking
    ba_buffer_df["ba_mask"] = 1
    #assgin "no burned area" with class 0
    ba_buffer_df["burned_bin"] = 0

    ##################################

    #concat burned area and no burned area dataframe and set crs
    ba_aoi_yr = pd.concat([ba_buffer_df.to_crs(crs),ba_aoi_yr_ba.to_crs(crs)],ignore_index=True)

    ### prepare dataframe for rasterizing
    #convert first_seen attribute to datetime
    ba_aoi_yr["first_seen_dt"] = pd.to_datetime(ba_aoi_yr["first_seen"])

    ######
    #group polygons 12:00 DOY - 12:00 DOY+1
    #weather parameter are selected before 12 am of day x and represent danger at day x afternoon   = timestamp x
    #first seen date => i want it to reprent wildfire that occured after 12 am at day x             = timestamp x

    #grouping of first seen times of burnt area polygon
    #first seen time < 12 am e.g. 10 am (first seen date of most wildfire first seen in europe)     = timestamp x - 1
    #first seen time > 12 am e.g. 4 pm                                                              = timestamp x

    #set DOY of group
    ba_aoi_yr.loc[ba_aoi_yr['first_seen_dt'].dt.hour <= 12, 'doy_group'] = ba_aoi_yr['first_seen_dt'].dt.dayofyear-1
    ba_aoi_yr.loc[ba_aoi_yr['first_seen_dt'].dt.hour > 12, 'doy_group'] = ba_aoi_yr['first_seen_dt'].dt.dayofyear
    #set timestamp (12:00) of group
    ba_aoi_yr.loc[ba_aoi_yr['first_seen_dt'].dt.hour <= 12, 'date_group'] = ba_aoi_yr['first_seen_dt'].dt.normalize() + pd.Timedelta('12:00:00') - pd.DateOffset(1)
    ba_aoi_yr.loc[ba_aoi_yr['first_seen_dt'].dt.hour > 12, 'date_group'] = ba_aoi_yr['first_seen_dt'].dt.normalize() + pd.Timedelta('12:00:00') 
    #set timestamp as index
    ba_aoi_yr = ba_aoi_yr.set_index(pd.to_datetime(ba_aoi_yr["date_group"]))
    # print(ba_aoi_yr.head(1))

    ######
    #group dataframe by doy_group
    doy_groups = ba_aoi_yr.groupby("doy_group") #, "burn_severity", "confidence_total", "area_m", "date", "date_index_id"])
    #create list with dataframes of each group
    list_df_groups = [doy_groups.get_group(x) for x in doy_groups.groups]
    #rasterize geodataframe
    xr_ba_doy = []
    for df_doy in list_df_groups:
        grid = rasterize_burnedarea_doy(df_doy, s3_dummy)
        xr_ba_doy.append(grid)
        
    xr_ba_year = xr.combine_by_coords(xr_ba_doy)
    #resample to grid template for dataprocessing
    xr_ba_year_rs = xr_ba_year.rio.reproject_match(s3_dummy)

    #write clipped and rasterized burned area
    path_out = f"{path_output}\\burnedarea_mask_{area}_{year}.nc"
    print(f"writing clipped and rasterized burned area mask of {area} {year} to {path_out}")
    xr_ba_year_rs.to_netcdf(path_out)


print("done.")