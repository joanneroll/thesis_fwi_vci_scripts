#########################################
#clip_stack_era5_indices.py

#clipping global era5 fwi indices to europe
#stacking fwi indices yearwise (2016 - 2022)

#author: johanna roll
#2022
##########################################<

from os import listdir
from os.path import isfile, join
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

#folder for clipped data
data_output = r"D:\roll_thesis\data_fwi_vi\output\clipped_era5"

## data paths
#folder with historical indices folder, based on era5 = data to be resampled
datapath_historical_indices_folder = "D:\\roll_thesis\\data_fwi_vi\\input\\era5_historical_indices"

area = "andalucia"
europe_bounds = (-25.125, 29.875, 50.125, 71.625) #lsm mask extent

#bounds of area of interest 
#defined larger than actual AOI due to difference in spatial resolution
clip_bounds = (-8.50, 39.50, -1.00, 35.50)

############################
#list with available dates in folder
list_dates = []
for file in listdir(datapath_historical_indices_folder):
    if file.endswith(".nc"):
        # file_path = join(datapath_historical_indices_folder, file)
        file_split = file.split("_")
        # file_name = "_".join(file_split[1:-4])
        date = file_split[-5]
        if date not in list_dates:
            list_dates.append(date)

#region clip # needs transformation of x axis
geodf = gpd.GeoDataFrame(
    geometry=[
        box(clip_bounds[0]+360,clip_bounds[1],clip_bounds[2]+360,clip_bounds[3])
    ],
    crs="EPSG:4326"
)

###############################################
print("starting loading, clipping and stacking of era5 fwi indices")

#list for dataset containing indices per daily entry
list_indices_per_date = []
#list for indices per year - for clearing list / saving memory
list_indices_per_year = []

year_processed = list_dates[0][:4]
print(f"processing year: {year_processed}")

#iterate over available dates
for date in list_dates:
    #data paths to variables
    path_bui = f"{datapath_historical_indices_folder}\\ECMWF_FWI_BUI_{date}_1200_hr_v4.0_con.nc"
    path_dc = f"{datapath_historical_indices_folder}\\ECMWF_FWI_DC_{date}_1200_hr_v4.0_con.nc"
    path_dmc = f"{datapath_historical_indices_folder}\\ECMWF_FWI_DMC_{date}_1200_hr_v4.0_con.nc"
    path_ffmc = f"{datapath_historical_indices_folder}\\ECMWF_FWI_FFMC_{date}_1200_hr_v4.0_con.nc"
    path_fwi = f"{datapath_historical_indices_folder}\\ECMWF_FWI_FWI_{date}_1200_hr_v4.0_con.nc"
    path_isi = f"{datapath_historical_indices_folder}\\ECMWF_FWI_ISI_{date}_1200_hr_v4.0_con.nc"
    
    #load indivitual historical indices for sample day + write epsg 4326 as crs
    BUI = xr.load_dataset(path_bui).rio.write_crs(4326)
    DC = xr.load_dataset(path_dc).rio.write_crs(4326)
    DMC = xr.load_dataset(path_dmc).rio.write_crs(4326)
    FFMC = xr.load_dataset(path_ffmc).rio.write_crs(4326)
    FWI = xr.load_dataset(path_fwi).rio.write_crs(4326)
    ISI = xr.load_dataset(path_isi).rio.write_crs(4326)

    #collect historical indices
    #keys: filename, data variable name
    dict_era5 = {(f"BUI_{date}", "bui"): BUI,
                (f"FFMC_{date}", "ffmc"): FFMC,
                (f"DC_{date}", "dc"): DC,
                (f"DMC_{date}","dmc"): DMC,
                (f"FWI_{date}","fwi"): FWI,
                (f"ISI_{date}","isi"): ISI}
    
    #get year from datestring
    year = date[:4]

    #keep track on processing
    #trigger print statement for new year being processed
    #merge indices per year
    if not year == year_processed:
        #merge indices per year and collect in list
        indices_per_year = xr.merge(list_indices_per_date)

        #save indices for year
        file_name = f"{data_output}\indices_{area}_dataset_{year_processed}.nc"

        indices_per_year.to_netcdf(file_name)

        # list_indices_per_year.append(indices_per_year)
        #clear list containing daily indices of processed year
        list_indices_per_date = []

        #pass year in variable and print statement for progress update
        year_processed = year
        print(f"processing year: {year_processed}")

    #list for collecting dataarrays of day
    data_list = []

    for name, data in dict_era5.items():
        #access data variable of raster 
        #e.g. for FWI raster: FWI_FWI_20200714.fwi
        data_variable = getattr(data, name[1])

        # # export raster individually as geotiff
        # data_variable.rio.to_raster(path_output)

        #collect clipped data in list for exporting "stacked" datavariables
        data_list.append(data_variable)

    #combine clipped dataarrays (fwi,bui,isi,ffmc,dmc,dc) in dataset
    dataset = xr.combine_by_coords(data_list)

    #clip to andalucia (bounding box)
    dataset_clipped = dataset.rio.clip(geodf.geometry, geodf.crs)

    #add dataset to list
    list_indices_per_date.append(dataset_clipped) 

#merge indices of last processed year and append to list
indices_per_year = xr.merge(list_indices_per_date)

#export as netcdf
file_name = f"{data_output}\indices_{area}_dataset_{year_processed}.nc"
indices_per_year.to_netcdf(file_name)

# list_indices_per_year.append(indices_per_year)

# #merge all datasets    
# all_indices = xr.merge(list_indices_per_year)

# dataset_out = f"{data_output}\indices_{area}_dataset_{list_dates[0]}_{list_dates[-1]}.nc"
# all_indices.to_netcdf(dataset_out)

print("done.")
