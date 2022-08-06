#########################################
#test_param_newfiw.py

#run validation of fire danger based on burned area
#era5land and era5 original fwi
#parameter tests for integrating the vci
#run validation on tested adapted fire danger index
#outputs saved per month

#author: johanna roll
#2022
##########################################

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import pickle
from datetime import datetime

import aux_functions
import analysis_functions

#path = r"D:\roll_thesis\data_fwi_vi\output\analysis_preproc"
# path = r"C:\Users\johan\Desktop\master_data\from_dlr\analysis_preproc"
path = r"C:\Users\rolljo\Desktop\masterdata\analysis_preproc"
path_out=f"{path}\\output"
if not os.path.exists(path_out):
    os.makedirs(path_out)

#set parameter
#crs of data
crs = 4326
#define parameter weights
weight = [5,4,3,2]

#months for analysis
months = [7,8,9,6]

# #years for analysis
# years = [2016,2017,2018,2019,2020,2021]

#define functions
def param_stats(fwi, fwi_plus,p_run,effect_run,vci_factor,ndvi_factor_n9=np.nan, ndvi_factor_n11=np.nan,ndvi_factor_act=False):   
    #calc stats
    fwi_mean = fwi.mean()
    fwi_new_mean = fwi_plus.mean()

    #variables for "run_analysis"
    dict_input = {
        "years_list" : f"{np.unique(ds.time.dt.year.values).min()}_{np.unique(ds.time.dt.year.values).max()}",
        "months_list": f"{np.unique(ds.time.dt.month.values).min()}_{np.unique(ds.time.dt.month.values).max()}",
        "vci_factor_mean" : np.round(vci_factor.mean().values,2), 
        "effect" : effect_run,
        "weight_p" : p_run,
        # "ndvi_factor_active" : ndvi_factor_act,
        # "ndvi_factor_n0.95" : ndvi_factor_n9,
        # "ndvi_factor_n1.05" : ndvi_factor_n11,
        "fwi_mean" : np.round(float(fwi_mean),2),
        "fwi_new_mean" : np.round(float(fwi_new_mean),2),
        "fwi_change_abs" : np.round(float(fwi_new_mean-fwi_mean),2),
        "fwi_change_perc" : np.round(float(fwi_new_mean/fwi_mean*100),2)
    }
    return dict_input


def extract_info(fwi_dict, fwi_str):
    param_of_interest = ['years_list', 'months_list', "vci_factor_mean", "effect", "weight_p","fwi_mean", "fwi_new_mean", "fwi_change_abs",
                         "fwi_change_perc", f"rmse_{fwi_str}_BA" , f"pearson_{fwi_str}_BA", f"mape_{fwi_str}_BA", f"pearson_NDVI_{fwi_str}", f"pearson_VCI_{fwi_str}"]
    df = pd.DataFrame(fwi_dict, columns = param_of_interest, index=[0])

    #add info metrics from rf classification
    df["area burned (hc)"] = fwi_dict["df_ba_classified"]["area_sev"][1.0]
    df["rf IoU class burned"] = fwi_dict["metrics_class_1burned"]["iou"]
    df["area not burned (hc)"] = fwi_dict["df_ba_classified"]["area_sev"][0.0] 
    df["rf IoU class not burned"] = fwi_dict["metrics_class_0noburned"]["iou"]

    df["metrics_class_1burned_iou"] = fwi_dict["metrics_class_1burned"]["iou"]
    df["metrics_class_iou"] = fwi_dict["iou"]
    df["metrics_class_0noburned_iou"] = fwi_dict["metrics_class_0noburned"]["iou"]

    df["metrics_class_1burned_acc"] = fwi_dict["metrics_class_1burned"]["acc"]
    df["metrics_class_0noburned_acc"] = fwi_dict["metrics_class_0noburned"]["acc"]
    
    df["metrics_class_acc"] = fwi_dict["acc"]

    return df

##############################

#collect all months in dataframe
df_collect = []
# df_collect_era5 = []

for month in months:
    print(f"testing parameter for month {month} from 2016-2021")
    print("time ",datetime.now().strftime("%H:%M:%S"))
    #define output folder
    path_output=f"{path_out}\\output_month{month}"
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    ##### preparation
    #load data
    print("load data")
    fwi_era5land = xr.load_dataset(f"{path}\\fwi_era5land\\fwi_era5land_month{month}.nc").rio.write_crs(crs)
    #small bugfix concerning timerange of fwi dataset
    fwi_era5land = fwi_era5land.sel(time=slice(("2016-01-01"),None))
    fwi_era5 = xr.load_dataset(f"{path}\\fwi_era5\\fwi_era5_month{month}.nc").rio.write_crs(crs)
    ba_mask_ds = xr.load_dataset(f"{path}\\burned_area\\burned_area_month{month}.nc").rio.write_crs(crs)
    ndvi = xr.load_dataset(f"{path}\\ndvi_smoothed\\ndvi_smoothed_month{month}.nc").rio.write_crs(crs)
    ndvi_dif = xr.load_dataset(f"{path}\\ndvi_difference\\ndvi_difference_month{month}.nc").rio.write_crs(crs)
    vci = xr.load_dataset(f"{path}\\vci\\vci_month{month}.nc").rio.write_crs(crs)                   

# for year in years:
#     print(f"testing parameter for year {year} in fireseason June - September")
#     print("time ",datetime.now().strftime("%H:%M:%S"))
#     #define output folder
#     path_output=f"{path_out}\\output_{year}"
#     if not os.path.exists(path_output):
#         os.makedirs(path_output)

#     ##### preparation
#     #load data
#     print("load data")
#     fwi_era5land = xr.load_dataset(f"{path}\\fwi_era5land\\fwi_era5land_{year}.nc").rio.write_crs(crs)
#     fwi_era5 = xr.load_dataset(f"{path}\\fwi_era5\\fwi_era5_{year}.nc").rio.write_crs(crs)
#     ba_mask_ds = xr.load_dataset(f"{path}\\burned_area\\burned_area_{year}.nc").rio.write_crs(crs)
#     ndvi = xr.load_dataset(f"{path}\\ndvi_smoothed\\ndvi_smoothed_{year}.nc").rio.write_crs(crs)
#     ndvi_dif = xr.load_dataset(f"{path}\\ndvi_difference\\ndvi_difference_{year}.nc").rio.write_crs(crs)
#     vci = xr.load_dataset(f"{path}\\vci\\vci_{year}.nc").rio.write_crs(crs)

    #ensure exact alignmnent of coordinate digits
    print("prepare dataset")
    #fwi
    fwi_era5land['y'] = np.round(fwi_era5land['y'], 6)
    fwi_era5land['x'] = np.round(fwi_era5land['x'], 6)

    #fwi era5
    fwi_era5['y'] = np.round(fwi_era5['y'], 6)
    fwi_era5['x'] = np.round(fwi_era5['x'], 6)

    #ba_mask
    ba_mask_ds['y'] = np.round(ba_mask_ds['y'], 6)
    ba_mask_ds['x'] = np.round(ba_mask_ds['x'], 6)

    #ndvi diff
    ndvi_dif['y'] = np.round(ndvi_dif['y'], 6)
    ndvi_dif['x'] = np.round(ndvi_dif['x'], 6)

    #vci
    vci['y'] = np.round(vci['y'], 6)
    vci['x'] = np.round(vci['x'], 6)

    #ndvi smoothed 30 days mosaic
    ndvi['y'] = np.round(ndvi['y'], 6)
    ndvi['x'] = np.round(ndvi['x'], 6)

    ##### 
    # prepare dataset
    # combine data in one dataset (further ensuring alignment and equal shape of data variables)
    ds = fwi_era5land.copy()
    ds["fwi_era5"] = fwi_era5.fwi
    ds["ba_mask"] = ba_mask_ds.burned_bin
    ds["ndvi_dif"] = ndvi_dif.ndvi
    ds["ndvi"] = ndvi.ndvi
    ds["vci"] = vci.vci
    #inverse vci
    ds["vci_inv"] = 1-vci.vci

    #free up RAM
    del fwi_era5land, ba_mask_ds, vci, ndvi, ndvi_dif, fwi_era5

    #round digits of dataset
    ds = np.round(ds,2)
    #transpose dimensions / ensure same order of dimension 
    ds = ds.transpose("time", "y", "x",...)

    #reproject ds to laea crs with 300m resolution
    print("resample dataset to crs 3035 (resolution = 300m)")
    ds = ds.rio.reproject(dst_crs=3035, resolution=300, nodata=np.nan)

    #chunk ds for paralell computing
    ds = ds.chunk({"x": 200, "y": 200})#.load()

    print("dataset prepared")

    #era5 land collections
    df_collect = []
    dict_collect = {}

    #era5 collections
    df_collect_era5 = []
    dict_collect_era5 = {}

    ##############
    ### run with original fwi 
    print("run analysis with original fwi (era5-land)")
    print("time ",datetime.now().strftime("%H:%M:%S"))
    #param dictionary    
    dict_input = param_stats(ds.fwi, ds.fwi,"None","None", xr.DataArray(1))
    #analyse original fwi
    dict_merge_fwi = analysis_functions.run_analysis(ds.fwi,ds.ba_mask,ds.ndvi_dif,ds.ndvi,ds.vci,dict_input,path_output=path_output,info_b=True) #show_plot_b=True,info_b=True
    #extract relevant info
    df_fwi = extract_info(dict_merge_fwi, fwi_str="FWI")
    #save in lists
    df_collect.append(df_fwi)
    #save dict in dict with parameters as key
    key = f"e{dict_merge_fwi['effect']}_p{dict_merge_fwi['weight_p']}" #_nf{str(dict_merge_fwi['ndvi_factor_active'])}"
    dict_collect[key] = dict_merge_fwi

    print("run analysis with original fwi (era5)")
    print("time ",datetime.now().strftime("%H:%M:%S"))
    #param dictionary    
    dict_input = param_stats(ds.fwi_era5, ds.fwi,"None", "None", xr.DataArray(1))
    #analyse original fwi
    dict_merge_fwi = analysis_functions.run_analysis(ds.fwi_era5,ds.ba_mask,ds.ndvi_dif,ds.ndvi,ds.vci,dict_input,path_output=path_output,fwi_type="FWI_ERA5") #show_plot_b=True,info_b=True
    #extract relevant info
    df_fwi = extract_info(dict_merge_fwi,fwi_str="FWI_ERA5")
    #save in lists
    df_collect_era5.append(df_fwi)
    #save dict in dict with parameters as key
    key = f"e{dict_merge_fwi['effect']}_p{dict_merge_fwi['weight_p']}" #_nf{str(dict_merge_fwi['ndvi_factor_active'])}"
    dict_collect_era5[key] = dict_merge_fwi

    #prepare variables for parameter test of timeseries
    print("prepare variables for parameter test")

    #distance of vci value to 1
    vci_dist = 1 - ds.vci_inv

    #run parameter tests on FWI ERA5 Land and ERA5 respectively
    for fwidata,fwitype,fwidf,fwidict in zip([ds.fwi, ds.fwi_era5],["FWI","FWI_ERA5"],[df_collect,df_collect_era5],[dict_collect,dict_collect_era5]):
    # for fwidata,fwitype,fwidf,fwidict in zip([ds.fwi],["FWI"],[df_collect],[dict_collect]):
        print(f"parameter test for {fwitype}")
        print("time ",datetime.now().strftime("%H:%M:%S"))

        #set variables for progress bar
        counter = 0
        param_runs = 8

        for p in weight:
            print(f"testing vci weight 1/{p}")
            print("time decreasing ",datetime.now().strftime("%H:%M:%S"))
            #DECREASING fwi
            effect = "-"
            
            vci_p = vci_dist/p
            ### ======>>> Calc VCI Factor #1
            vci_factor = np.round((ds.vci_inv + vci_p),2)
            #approach 2
            # vci_factor = np.round((1 - vci_p),2)

            #calc new fwi
            fwi_plus = fwidata*vci_factor
            
            #param dictionary    
            dict_input = param_stats(fwidata, fwi_plus,p,effect,vci_factor)
            
            #analyse new fwi
            dict_merge_newfwi = analysis_functions.run_analysis(fwi_plus,ds.ba_mask,ds.ndvi_dif,ds.ndvi,ds.vci,dict_input,path_output=path_output,fwi_type=fwitype) #show_plot_b=True,info_b=True
            #extract relevant info
            df_newfwi = extract_info(dict_merge_newfwi, fwi_str=fwitype)
            #save in lists
            fwidf.append(df_newfwi)
            #save dict in dict with parameters as key
            key = f"e{dict_merge_newfwi['effect']}_p{dict_merge_newfwi['weight_p']}"
            fwidict[key] = dict_merge_newfwi
            
            #print progress    
            counter += 1
            aux_functions.progress(counter, param_runs)
            
            
            ############################################################
            ###################
            ## INCREASING fwi
            effect = "+"
            print("time increasing fwi",datetime.now().strftime("%H:%M:%S"))

            vci_p = vci_dist/p
            ### ======>>> Calc VCI Factor #3
            vci_factor = np.round((1 + vci_dist - vci_p),2)
            # approach 2
            # vci_factor = np.round((1 + vci_p),2)
                
            #calc new fwi
            fwi_plus = fwidata*vci_factor
            
            #param dictionary    
            dict_input = param_stats(fwidata, fwi_plus,p,effect,vci_factor)
            
            #analyse new fwi
            dict_merge_newfwi = analysis_functions.run_analysis(fwi_plus,ds.ba_mask,ds.ndvi_dif,ds.ndvi,ds.vci,dict_input,path_output=path_output,fwi_type=fwitype) #show_plot_b=True,info_b=True
            #extract relevant info
            df_newfwi = extract_info(dict_merge_newfwi, fwi_str=fwitype)
            #save in lists
            fwidf.append(df_newfwi)
            #save dict in dict with parameters as key
            key = f"e{dict_merge_newfwi['effect']}_p{dict_merge_newfwi['weight_p']}"
            fwidict[key] = dict_merge_newfwi

            #print progress    
            counter += 1
            aux_functions.progress(counter, param_runs)
            
            #############

    #monthly output
    #save dictionaries
    dict_out = open(f"{path_output}\\dict_fwi_month{month}.pickle", "wb")
    pickle.dump(dict_collect, dict_out)
    dict_out = open(f"{path_output}\\dict_fwi_era5_month{month}.pickle", "wb")
    pickle.dump(dict_collect_era5, dict_out)

    #concat dataframes 
    df_ts = pd.concat(df_collect,ignore_index=True)
    df_tsera5 = pd.concat(df_collect_era5,ignore_index=True)

    #save dataframes to csv
    df_ts.to_csv(f"{path_output}\\df_fwi_month{month}.csv", index=True)
    df_tsera5.to_csv(f"{path_output}\\df_fwi_era5_month{month}.csv", index=True)

    #yearly output
    #save dataframes to csv
    # df_ts.to_csv(f"{path_output}\\df_fwi_{year}.csv", index=True)
    # df_tsera5.to_csv(f"{path_output}\\df_fwi_era5_{year}.csv", index=True)

    # #save dictionaries
    # dict_out = open(f"{path_output}\\dict_fwi_{year}.pickle", "wb")
    # pickle.dump(dict_collect, dict_out)
    # dict_out = open(f"{path_output}\\dict_fwi_era5_{year}.pickle", "wb")
    # pickle.dump(dict_collect_era5, dict_out)

#concat dataframes 
df_ts = pd.concat(df_collect,ignore_index=True)
df_tsera5 = pd.concat(df_collect_era5,ignore_index=True)

#save dataframes to csv
df_ts.to_csv(f"{path_out}\\df_fwi_months_total.csv", index=True)
df_tsera5.to_csv(f"{path_out}\\df_fwi_era5_months_total.csv", index=True)

print("done. ",datetime.now().strftime("%H:%M:%S"))