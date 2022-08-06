#########################################
#prep_input_param.py

##prepare input parameter for fwi calculation
#temperature, relative humidity, precipitation, wind speed

#author: johanna roll
#2022
##########################################

from os import listdir
from os.path import join

import xarray as xr
import numpy as np

import aux_functions

#directory with data for same region of interest but complementary time span (subject to change)
path_data = "D:\\roll_thesis\\data_fwi_vi"

datapath = f"{path_data}\\input\\cds_era5land_weatherparam_andalucia_05_2015_12_2021"
path_output = f"{path_data}\\output\\indices_calculated"

#area
area="andalucia"


##################################
#set time noon
noon_threshold = "12:00:00" # UTC time 
#set afternoon threshold for 24hrs total precipitation calculation
afternoon_threshold = "13:00:00" # 

##################################

#list with available dates in folder
dataarrays = []

#load grib data and merge dataset (along dimension "time")
for file in listdir(datapath):
    if file.endswith(".grib"):
        file_path = join(datapath, file)

        print(f"loading {file}")
        
        array = xr.load_dataset(file_path, engine="cfgrib").rio.write_crs(4326)
        dataarrays.append(array)

ds = xr.merge(dataarrays)

print("merging datasets done")

'''
filter required xarrays from data & convert units: 
- temperature 2m above surface (t2m) in °C
- relative humidty in % (calcualted from temperature and dew point (d2m))
- windspeed in km/h (calcualted from windcomponents u (u10) and v (v10))
--> time of observation: noon
--> given issue of first data entries (stated above), the variables/dataarrays only taking value measured at noon will indexed to second day of complete daily data entries

- total precipitation over 24 hours in mm (calculated from hourly total precipitation (tp))
--> to ensure sensible calculation of total rainfall, dataarray will be indexed to first day of complete daily data entry (first day will be dropped afterwards to ensure same time dimension of all variables used)
'''

#isolate relevant weather data at noon (12:00:00 UTM) and convert units accordingly
#downloaded data overall will be trimmed starting with second daily data entry with complete 24/7 measurements
#temperature (kelvin -> celsius)
t2m_noon = ds.sel(step=noon_threshold).t2m-273.15
t2m_noon = t2m_noon[2:]
t2m_noon = t2m_noon.rename(new_name_or_name_dict = "t2m_noon")

#dewpoint (kelvin -> celcius)
d2m_noon = ds.sel(step=noon_threshold).d2m-273.15
d2m_noon = d2m_noon[2:]

#windcomponent u (m/s -> km/h)
u10_noon = ds.sel(step=noon_threshold).u10*3.6
u10_noon = u10_noon[2:]

#windcomponent v (m/s -> km/h)
v10_noon = ds.sel(step=noon_threshold).v10*3.6
v10_noon = v10_noon[2:]

#precipitation (m (meter water equivalent) -> mm)
#start data from first day of complete daily data for sensible calculation of accumulated rainfall
tp_hourly = ds.tp[1:]*100


#######################
##calculation
#relative humidity
#calcualtion based on magnus formula with approximation as recommended by Alduchov and Eskridge (1996)
rh_noon = aux_functions.magnus_formula(t2m_noon,d2m_noon)
#set metadata
rh_noon.name = "rh_noon"
rh_noon.attrs["units"] = "%"
rh_noon.attrs["description"] = "relative humidity"

#windspeed
#calculated from u and v wind component in 10m above ground (see: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398)
ws10_noon = np.sqrt(np.square(u10_noon)+np.square(v10_noon))
#set metadata 
ws10_noon.name = "ws10_noon"
ws10_noon.attrs["units"] = "km/h"
ws10_noon.attrs["description"] = "windspeed"

#precipitation
tp_noon = aux_functions.calc_tp_noon(tp_hourly, noon_threshold, afternoon_threshold)
#set metadata
tp_noon.name = "tp_noon"
tp_noon.attrs["tp_noon"] = "tp_noon"
tp_noon.attrs["description"] = "24hrs precipitation"

#combine input parameter data
input_param = xr.merge([t2m_noon,rh_noon, ws10_noon,tp_noon])

#export input parameter to netcdf
date_min = input_param.time.min().values
date_max = input_param.time.max().values

path_output_input_param_netcdf = f"{path_output}\\input_param_{area}_{str(date_min)[:10]}_{str(date_max)[:10]}.nc"
input_param.to_netcdf(path_output_input_param_netcdf)
print("collecting input parameter done.")
print(f"writing input parameter to file {path_output_input_param_netcdf}.")