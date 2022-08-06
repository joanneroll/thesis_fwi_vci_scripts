#########################################
#aux_functions.py

#auxiliary functions for fwi calculation and resampling of datasets

# apply burned area mask on ndvi difference (magnus_formula)
# calculate area in hectar from number of pixel (calc_windspeed)
# print_raster (print georeference of raster/dataarray/set)
# norm_dataarray (norm dataarray with alignment of coordinate digits)
# retrieve_name (aux function for printing variables names during process)
# check_raster_georeference (compare georefrence for two dataarrays/sets)
# check_fix_raster_resolution (compare georefrence for two dataarrays/sets and set resolution if necessary)
# check_fix_raster_georeference (compare georefrence for two dataarrays/sets and resample/reproject/clip if necessary)
# sync_time_range (ensure same tstart and tend of two datasets/arrays)
# check_fix_digits_alignment 
# progress (progress bar in terminal)


#author: johanna roll
#2022
##########################################

# packages
import sys
import numpy as np
import xarray as xr
import inspect

def magnus_formula(T,DT,a=17.627,b=243.04,c=6.1094):
    '''
    calculation of relative humidity
    approximation, if not other specified, calculated based on recommended values by Alduchov and Eskridge (1996)
    
    input parameters:
    temperature (T) in °C (xarray dataarray)
    dewpoint (DT) in °C (xarray dataarray)
    
    a - constant
    b in °C
    c in hPa
    
    returns: relative humidity in %
    '''
    #actual saturation vapor pressure (dew point)
    e = c * np.exp((a * DT)/(DT + b))
    #possible saturation vapor pressure (temperature)
    e_s = c * np.exp((a * T)/(T + b)) 
    
    return 100*(e/e_s)

def calc_windspeed(u_component, v_component):
    '''
    calculation of windspeed based on wind components

    further info: https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398

    input parameters:
    u component (xarray dataarray)
    v component (xarray dataarray)

    returns: windspeed (in unit of windcompontents)
    '''
    
    ws = np.sqrt(np.square(u_component)+np.square(v_component))
    
    return ws

def calc_tp_noon(tp_hourly, noon_threshold, afternoon_threshold):
    '''
    calculation of total precipitation over 24 hours with noon as threshold
    based on ERA5 land tp data with hourly total precipitation

    for sensible FWI calculation: unit of precipitation -> mm

    input parameters:
    hourly total precipitation (xarray dataarray)
            dimensions of tp_hourly: ('time', 'step', 'latitude', 'longitude')
    threshold noon (end of 24 hrs accumulation) (str)
    threshold afternoon (start of 24 hrs accumulation) (str)
    
    '''
    #precipitation afternoon of first complete daily data entry
    tp_hourly_til_noon = tp_hourly.isel(time=0).loc[afternoon_threshold:]
    p_afternoon = tp_hourly_til_noon.sum(dim="step",min_count=1) #min_count=1 -> minimum one non-nan value for performing the operation

    #list for collecting accumulated precipitation
    list_tp_acc = []

    for day in tp_hourly: #[:,]:
        # print(day)
        #precipitation morning until noon (of day of interest)
        #looping over dimension "step"
        day_time = day.time
        # print(day_time)
        p_morning = day.loc[:noon_threshold,...].sum(dim="step",min_count=1) #min_count=1 -> minimum one non-nan value for performing the operation
        # print("p_morning \n", p_morning)

        #accumulated precipitation at noon in the previous 24h
        acc_precip_noon = p_afternoon + p_morning #acc_precip_noon datatype dataarray
        acc_precip_noon["time"]=day_time
        # print("acc_precip_noon \n", acc_precip_noon)

        #set time as dimension
        acc_precip_noon = acc_precip_noon.expand_dims("time")

        #add accumulated rainfall and date to lists
        list_tp_acc.append(acc_precip_noon)

        #precipitation after noon (counts for accumulated rainfall for next day)
        #selecting 13:00.00 - 00:00:00 and accumulating precipitation
        p_afternoon = day.loc[afternoon_threshold:,...].sum(dim="step",min_count=1)
        # print("p_afternoon \n", p_afternoon, "\n")

    #concatenate dataarrays in list
    tp_noon = xr.concat(list_tp_acc, dim="time")
    #slice tp_noon to start at day with second complete daily entry, analog to other dataarrays
    tp_noon = tp_noon[1:]
    
    return tp_noon


def dataarray_from_fwiCodes(code_list, code_name, code_name_lng, dates_array, coord_surface, coord_lat, coord_lng, coord_step):
    '''
    function designed to create dataarrays from lists resulting from pyfwi calculation (fwi_main.py / calc_fwi.py)
    is used during cell-wise iteration over dataset to create a resulting dataarray for the respective cell
    
    parameter: 
    - code_list (returned parameter from fwi calculation) FWI calculation code when iterating over one grid cell over time (list)
    - code_name: name of calculated code/index (str)
    - code_name_lng: long name of calculated code/index (str)
    - dates_array: dataarray with timestamps for all entries in code_list (equal size required) (xarray dataarray)
    - coord_surface, coord_lat, coord_lng, coord_step: metadata collected during script / maybe change that later and include in function (float)
    
    
    returns: dataarray with corresponding dimension time
    '''
    #maybe incorporate check data len(code_list) and dates_array equals
    
    #isolite calculated codes
    dataarray = xr.DataArray(code_list)

    # #metadata from original dataarray from grid cell
    # coord_surface = ds_single.coords["surface"].values
    # coord_lat = ds_single.coords["latitude"].values
    # coord_lng = ds_single.coords["longitude"].values
    # coord_step = noon_threshold
    
    code_array = xr.DataArray(dataarray, 
                        dims=["time",],
                        coords={
                            "time": dates_array,
                            "step": coord_step,
                            "surface": coord_surface,
                            "latitude": coord_lat, 
                            "longitude": coord_lng},
                        attrs={
                            "name_lng": code_name_lng},
                        name=code_name,
                    )
    
    #set float 32 as dtype (saving memory <=> float46)
    code_array = code_array.astype(np.float32, copy=False)
    
    return code_array

def print_raster(raster):
    '''
    prints information on dataarray:
        - shape
        - resolution
        - bounds
        - crs
    '''

    print(
        f"shape: {raster.rio.shape}\n"
        f"resolution: {raster.rio.resolution()}\n"
        f"bounds: {raster.rio.bounds()}\n"
        f"CRS: {raster.rio.crs}\n"
    )

def norm_dataarray(dataarray):
    '''
    norm dataarray containing lat/lng information to ensure sensible computation between dataarrays
    
    transpose/norm order of lat/lng and time dimension if time dimension available
    rounds coordinate values to 3 digits
    
    input parameter:
    dataarray (with dimensions lag / lng (/ time) )
    
    returns: xr dataarray
    '''
    
    #name dimensions: longitude -> x, latitude -> y
    if not ("x" in dataarray.dims.keys()): #or ("y" in dataarray.dims.keys()):
        dataarray = dataarray.rename({"longitude": "x", "latitude": "y"})
        
    #order dimensions
    if "time" in dataarray.dims.keys():
        dataarray = dataarray.transpose('time', 'y', 'x', ...)
    
    #norm float tolerance of coordinate values to ensure exact alignment
    #round 3 digits (era5 (grib2) in microdegrees (5), era5 land (grib1) in milidegrees (3)
    dataarray['x'] = np.round(dataarray['x'], 3)
    dataarray['y'] = np.round(dataarray['y'], 3)
    
    return dataarray

def retrieve_name(var):
    '''
    get variable name as string within a function
    (https://stackoverflow.com/a/18425523/17984020)
    
    input parameter:
    variable
    
    returns:
    name of variable (str)
    '''
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    
    #get var name of scope earlier - remove one "f_back"
    #callers_local_vars = inspect.currentframe().f_back.f_locals.items()

   
    name = [var_name for var_name, var_val in callers_local_vars if var_val is var]

    # try:
    #     name[0]
    # except:
    #     callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    #     name = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    
    return name[0]

def check_raster_georeference(array1, array2, resample_option=None):
    '''
    check shape, resolution, bound and crs of two arrays
    
    #resample option could be included (dataarray1.rio.reproject_match(dataarray2))
    #for now simply print statements for information
    #combination of check_raster_georeference, check_fix_raster_resolution, check_fix_raster_georeference possible / advisable
    '''
 
    if array1.rio.shape == array2.rio.shape:
        print("shape equal", array1.rio.shape)
        
    else:
        print("!!! check shape - maybe resampling necessary")
        print(retrieve_name(array1), array1.rio.shape, retrieve_name(array2), array2.rio.shape)
        
    #check resolution 
    if array1.rio.resolution() == array2.rio.resolution():
        print("resolution equal", array1.rio.resolution())
        
    else:
        print("!!! check resolution - resampling necessary")
        print(retrieve_name(array1), array1.rio.resolution(), retrieve_name(array2), array2.rio.resolution())
    
    #check boudns
    if array1.rio.bounds() == array2.rio.bounds():
        print("bounds equal", array1.rio.bounds())
        
    else:
        print("!!! check bounds - resampling or rounding of digits necessary")
        print(retrieve_name(array1), array1.rio.bounds(), retrieve_name(array2), array2.rio.bounds())
        
    
    #check crs
    if array1.rio.crs == array2.rio.crs:
        print("crs equal", array1.rio.crs)
        
    else:
        print("!!! check crs - maybe resampling necessary")
        print(retrieve_name(array1), array1.rio.crs, retrieve_name(array2), array2.rio.crs)

def check_fix_raster_resolution(array, template, crs):
    """     
    check resolution of array with template array and resample if necessary
    small sister of 'check_fix_raster_georeferences' 

    input:
        - template array (xarray)
        - check array (xarray)
        - destination crs (int)
    returns:
        - resampled check array (if necessary)

    #combination of check_raster_georeference, check_fix_raster_resolution, check_fix_raster_georeference possible / advisable
    """
    if template.rio.resolution() != array.rio.resolution():
        print(f"resampling to crs {crs} and resolution {template.rio.resolution()}")
        array_rs = array.rio.reproject(dst_crs = crs, resolution = template.rio.resolution(), nodata=np.nan)

    return array_rs

def check_fix_raster_georeference(a1, a2, info = False):
    '''
    check shape, resolution, bound and crs of two arrays
    resampling a1 to a2 if georeference differs
    
    input:
        - check array (a1) (xarray)
        - template array (a2) (xarray)
        - info (bool):
            if true: prints print statements

    #combination of check_raster_georeference, check_fix_raster_resolution, check_fix_raster_georeference possible / advisable
    '''
    if (a1.rio.shape != a2.rio.shape) or (a1.rio.resolution() != a2.rio.resolution()) or (a1.rio.bounds() != a2.rio.bounds()) or (a1.rio.crs != a2.rio.crs):
        #print more info if parameter was set to True 
        if info == True:
            print("georeference of array differs (shape/resolution/bounds/crs) --> resampling/reprojecting array 1")

            print("georeference array 1")
            print_raster(a1)
            print("georeference array 2")
            print_raster(a2)

        a1_res = a1.rio.reproject_match(a2,nodata=np.nan) 
        return a1_res
    else:
        if info == True:
            print("exact georeference")
        return a1

def sync_time_range(a, b):
    '''
    comapare two dataarrays in terms of temporal resolution
    trimms the arrays accordingly to have to arrays with matching start and end date

    disclaimer: right now does not check missing dates between start and end date
    requires: same (hour) timestamp 

    input parameter:
    dataarray a with datetime dimension
    dataarray b with datetime dimension

    returns:
    trimmed dataarrays if necessary 
    '''
    print("----------------------")

    #get start and end dates of dataarrays
    start_a = a.time[0] #min(a.time)  
    end_a = a.time[-1]  #max(a.time) 

    start_b = b.time[0] #min(b.time) 
    end_b = b.time[-1] #max(b.time) 

    #start dates
    if start_a == start_b:
        print("start dates match:")
        #some issue with retrieve_name in some cases.. not sure
        try:
            print(retrieve_name(a), start_a.values, retrieve_name(b), start_b.values)
        except:
            print(start_a.values, start_b.values)
    else:
        print("start dates dont match:")
        
        #some issue with retrieve_name in some cases.. not sure
        try:
            print(retrieve_name(a), start_a.values, retrieve_name(b), start_b.values)
        except:
            print(start_a.values, start_b.values)

        
        #trimm data depending on which data starts later
        if start_a > start_b:
            b = b.where(b.time>= start_a, drop=True) #= min(a.time)
        elif start_a < start_b:
            a = a.where(a.time>= start_b, drop=True) #= min(a.time)
            
        print("--> trimming data (start date):")
        #some issue with retrieve_name in some cases.. not sure
        try:
            print(retrieve_name(a), a.time[0].values, retrieve_name(b), b.time[0].values)
        except:
            print(a.time[0].values, b.time[0].values)


    #end dates    
    if end_a == end_b:
        print("\nend dates match:")
        
        #some issue with retrieve_name in some cases.. not sure
        try:
            print(retrieve_name(a), end_a.values, retrieve_name(b), end_b.values)
        except:
            print(end_a.values, end_b.values)
    else:
        print("\nend dates dont match:")

        #some issue with retrieve_name in some cases.. not sure
        try:
            print(retrieve_name(a), end_a.values, retrieve_name(b), end_b.values)  
        except:      
            print(end_a.values, end_b.values)

        
        #trimm data depending on which data end earlier
        if end_a > end_b:
            a = a.where(a.time <= end_b, drop=True)
            
        elif end_a < end_b:
            b = b.where(b.time <= end_a, drop=True)
                    
        print("--> trimming data (end date):")
        
        #some issue with retrieve_name in some cases.. not sure
        try:
            print(retrieve_name(a), a.time[-1].values, retrieve_name(b), b.time[-1].values)    
        except:
            print(a.time[-1].values, b.time[-1].values)

    print("----------------------")

    
    return a, b

def check_fix_digits_alignment(a,b):
    '''
    checks alignment of two datarrays with dimensions containing x/y respectively lat/lng coordinates (float)
    rounds float equally to 3 digits

    prints further information on dataarrays if still not aligned after rounding of float

    exact alignment of arrays is required e.g. for the dataarray.where function to work (returns empty otherwise)

    input parameter:
    dataarray a with x/y (lat(lng)) dimension (float)
    dataarray b with x/y (lat(lng)) dimension (float)

    returns:
    dataarrays with rounded x/y digits if necessary
    '''

    try:
        xr.align(a,b, join="exact")
        print("arrays align")
    except ValueError as err:    
        # a['x'] = np.round(a['x'], 3)
        # a['y'] = np.round(a['y'], 3)

        # b['x'] = np.round(b['x'], 3)
        # b['y'] = np.round(b['y'], 3)
        
        a = norm_dataarray(a)
        b = norm_dataarray(b)
        
        print("arrays don't align --> round x and y coordinates (float) to 3 digits")
        
        try:
            xr.align(a,b, join="exact")
            print("arrays align")
        except:
            print("still not aligned, check time stamps or raster georeference (e.g. resampling required")
            check_raster_georeference(a,b)

    return a,b

def progress(count, total):
    '''
    progress bar

    adapted from: Copyright (c) 2016 Vladimir Ignatev "progress.py"
    '''

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100 * count / float(total), 5)
    bar = '█' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write(f"| {bar} | {percents} % complete\r")
    sys.stdout.flush()
        