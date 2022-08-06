#########################################
#fwi_main.py

#calculate fwi from prepared fire weather parameters
#code implemented from Wang et al. (2015)

#author: johanna roll
#2022
##########################################

import xarray as xr
import cfgrib
import numpy as np
import pandas as pd
import sys

from fwi_wang import FWICLASS
import aux_functions

#set variables
#path input parameter (script prep_input_param.py)
path_input_param_netcdf = r"C:\Users\johan\Desktop\master_data\data\output\indices_calculated\input_param_andalucia_2015-05-02_2021-12-31.nc"

path_output = "C:\\Users\\johan\\Desktop\\master_data\\data\\output\\indices_calculated"

#area
area="andalucia"
#set time noon
noon_threshold = "12:00:00" # UTC time 

######################################################
#fwi calculation
#load input parameter
input_param = xr.load_dataset(path_input_param_netcdf)

date_min = input_param.time.min().values
date_max = input_param.time.max().values

#iterate over individual grid cells
indices_lat_lng_totalarea = []

lat_length = input_param.sizes["latitude"]
lng_length = input_param.sizes["longitude"]

#initialize variables for progress bar
cells_total = lat_length*lng_length
counter = 1

print("starting fwi calculation\n")

for lat_idx in range(0,lat_length):
    for lng_idx in range(0,lng_length):

        #iterate over each grid cell
        # print(f"iterating over {lat_idx},{lng_idx}")
        input_param_lat_lng = input_param.isel(latitude=lat_idx, longitude=lng_idx)

        #print progress
        aux_functions.progress(counter, cells_total)
        counter += 1
        
        #collect some meta data information from single grid point for creating data
        coord_surface = input_param_lat_lng.coords["surface"].values
        coord_lat = input_param_lat_lng.coords["latitude"].values
        coord_lng = input_param_lat_lng.coords["longitude"].values
        coord_step = noon_threshold

        #convert np datetime64 ns to datetime
        time_dt = pd.to_datetime(input_param_lat_lng.time.values)
        #set hours 12 
        dates = time_dt.map(lambda t: t.replace(hour=12))
        
        #run fwi computation only if grid cell has values (is not nan)
        nan_check = np.isnan(input_param_lat_lng.t2m_noon)
        if np.any(nan_check) == False:
            # print(coord_lat,coord_lng)

            #get values from input parameters
            t_totaltime = input_param_lat_lng.t2m_noon.values
            rh_totaltime = input_param_lat_lng.rh_noon.values
            ws_totaltime = input_param_lat_lng.ws10_noon.values
            tp_totaltime = input_param_lat_lng.tp_noon.values

            #define start values: Lawson & Armitage, 2008, p. 14
            ffmc0 = 85
            dmc0 = 6
            dc0 = 15

            #lists for collecting calculated codes/indices for each cell over time range
            list_ffmc = []
            list_dmc = []
            list_dc = []
            list_isi = []
            list_bui = []
            list_fwi = []

            #loop over each day for year/time range
            for t,rh,ws,tp,month in zip(t_totaltime,rh_totaltime, ws_totaltime, tp_totaltime, dates.month.values):

                if rh>100.0:
                    rh = 100.0  

                #create FWI CLASS
                fwi_system = FWICLASS(t,rh,ws,tp)

                #calculate codes
                ffmc = fwi_system.FFMCcalc(ffmc0)
                dmc = fwi_system.DMCcalc(dmc0,month)
                dc = fwi_system.DCcalc(dc0,month)
                isi = fwi_system.ISIcalc(ffmc)
                bui = fwi_system.BUIcalc(dmc,dc)
                fwi = fwi_system.FWIcalc(isi,bui)

                #add codes to list
                list_ffmc.append(ffmc)
                list_dmc.append(dmc)
                list_dc.append(dc)
                list_isi.append(isi)
                list_bui.append(bui)
                list_fwi.append(fwi)

                #pass codes for next iteration
                ffmc0 = ffmc
                dmc0 = dmc
                dc0 = dc 
        
        #if no values is cell, set nan values for each daily entry
        #ensures that all grid cells are being kept
        # elif np.any(nan_check) == True: 
        else:

            # print("data missing, check input")
            # print(coord_lat, coord_lng)
            
            list_nan = len(dates) * [np.NaN]
            list_ffmc = list_dmc = list_dc = list_isi = list_bui = list_fwi = list_nan


        #calculation for time span done for grid cell
        #create dataarrays from lists
        ffmc_array = aux_functions.dataarray_from_fwiCodes(list_ffmc, "ffmc", "Fine Fuel Moisture Code", dates, coord_surface, coord_lat, coord_lng, coord_step)
        dmc_array = aux_functions.dataarray_from_fwiCodes(list_dmc, "dmc", "Duff Moisture Code", dates, coord_surface, coord_lat, coord_lng, coord_step)
        dc_array = aux_functions.dataarray_from_fwiCodes(list_dc, "dc", "Drought Code", dates, coord_surface, coord_lat, coord_lng, coord_step)
        isi_array = aux_functions.dataarray_from_fwiCodes(list_isi, "isi", "Initial Spread Index", dates, coord_surface, coord_lat, coord_lng, coord_step)
        bui_array = aux_functions.dataarray_from_fwiCodes(list_bui, "bui", "Build Up Index", dates, coord_surface, coord_lat, coord_lng, coord_step)
        fwi_array = aux_functions.dataarray_from_fwiCodes(list_fwi, "fwi", "Fire Weather Index", dates, coord_surface, coord_lat, coord_lng, coord_step)

        #merge calculated indices to dataset
        indices_lat_lng = xr.merge([ffmc_array, dmc_array, dc_array, isi_array, bui_array, fwi_array])
        #expand lat and lng dim
        indices_lat_lng = indices_lat_lng.expand_dims(["latitude", "longitude"])
    
        #collect datasets
        indices_lat_lng_totalarea.append(indices_lat_lng)        

#combine arrays
indices_area = xr.combine_by_coords(indices_lat_lng_totalarea)
path_output_indices_netcdf = f"{path_output}\\fwi_indices_{area}_{str(date_min)[:10]}_{str(date_max)[:10]}.nc"
indices_area.to_netcdf(path_output_indices_netcdf)

print("calculation of indices done.")
print(f"writing input parameter to file {path_output_indices_netcdf}.")