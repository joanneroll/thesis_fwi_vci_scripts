##########################################
#analysis_functions.py

#functions written for various analysis steps for validation 

#apply burned area mask on ndvi difference (apply_ba_mask)
#calculate area in hectar from number of pixel (calc_area)
#calculate area of burn severity for each fire danger class (calculate_ba_fwi_stats) - not included in final analysis
#plotting of linear regression and collect infos in dictionary (lin_reg_firedanger_dict)
#calcualting pearsons r and collect infos in dictionary (lin_reg_pearson_dict)
#random forest classification of burned area based on fire danger and collect infos in dictionary (predict_valid_firedanger)
#all functions called for running the analyis and collect infos in dictionary (run_analysis)

#author: johanna roll
#2022
##########################################

# function for analysis
import xarray as xr
import geopandas as gpd
import rioxarray
import numpy as np
import matplotlib.pyplot as plt
import aux_functions
import dask.array as da
import dask

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error,ConfusionMatrixDisplay,confusion_matrix,accuracy_score
from sklearn.feature_selection import r_regression,f_regression
from sklearn.ensemble import RandomForestClassifier

import ukis_metrics.seg_metrics as segm

#avoid creating large chunks during parallelized computation
dask.config.set({"array.slicing.split_large_chunks": True})

def apply_ba_mask(fwi,ba_mask,ndvi_difference,info_a=False):
    #apply burned area mask (dataarray) on ndvi difference data (dataarray) 

    #select fwi by available burned area dates
    #ensure variable name to be set to "fwi" - for consequitive functions to work properly
    fwi = fwi.rename(new_name_or_name_dict = "fwi")

    #select fwi entries by timestamps appearing in burned area data (ensure equal dimension shapes)
    fwi_burndates = fwi.sel(time=ba_mask.time)
   
    #mask ndvi difference data (== burn severity) by known/detected burned areas (processed by DLR)
    #all burned and no burned areas pos+neg
    # burnedarea_pn = ndvi_difference.ndvi.where(ba_mask >= 0) #when dataset
    burnedarea_pn = ndvi_difference.where(ba_mask >= 0) 
    #only burned areas 
    # burnedarea_p = ndvi_difference.ndvi.where(ba_mask == 1) #when dataset
    burnedarea_p = ndvi_difference.where(ba_mask == 1) 

    #mask binarized burned area by available ndvi difference values for achieving comparable output
    #background: given the preprocessing, the ndvi difference dataset may have nan values where the burned area dataset from DLR is valid
    # burnedarea_binarized = ba_mask.where(ndvi_difference.ndvi != np.nan) #when dataset
    burnedarea_binarized = ba_mask.where(ndvi_difference != np.nan)

    return fwi_burndates,burnedarea_pn,burnedarea_p,burnedarea_binarized


def calc_area(x, res = 300, unit="ha"):
    '''
    calculate area by multiplying number of pixel * resolution pixel
    crs should be projected for precise measurements
    '''
    #set dtype float32 to prevent long_scalar overflow
    x = x.astype(np.float32)

    if unit == "ha":
        a = (x*res*res/10000)
    elif unit == "m2":
        a = x*res*res
    elif unit == "km2":
        a = x*res*res/1000000
    
    a = np.round(a,2)
    return a

def calculate_ba_fwi_stats(fwi,ba,res=300,binarized=False,info=True):
    '''
    input: dataarray 
        - fireindex (dataarray)
        - burned area: 
            if binarized = False: masked ndvi_diff resp. burn_severity (dataarray) e.g. burned_area_p/burned_area_pn from def apply_bamask --> burned area with burned severity
            if binarized = True: binarized ba_mask (ba_mask.burned_bin) (dataarray) --> burned area binarized in no burned and burned areas
        - info (bool):
            if true: prints print statements
    returns: dataframe
        - area per burn severity
        - area per fwi class
    '''

    ## SEVERITY
    #total area burned by burn severity
    if binarized == False:
        #group burned area by severity
        if info == True:
            print("grouping burned area by burn severity")
        #create bins for burn_severity
        bin_sev = list(np.arange(-0.4, 0.1, 0.05))
        bin_sev = [round(x,2) for x in bin_sev]
        ba_sev_group = ba.groupby_bins(ba, bin_sev)
        #count ba severity bins pixel per group and calculate area
        #depending whether using parallelised computing
        try:
            area_per_sev_bin = xr.apply_ufunc(calc_area,ba_sev_group.count())
        except:
            area_per_sev_bin = xr.apply_ufunc(calc_area,ba_sev_group.count(),dask="parallelized")

    else:
        if info == True:
            print("grouping burned area by true burned and true no burned areas")
        #group burned area for burned areas (ba_mask.burned_bin == 1) and no burned areas (ba_mask.burned_bin == 0)
        ba_sev_group = ba.groupby(ba)
        #depending whether using parallelised computing
        try: 
            area_per_sev_bin = xr.apply_ufunc(calc_area,ba_sev_group.count())
        except:
            area_per_sev_bin = xr.apply_ufunc(calc_area,ba_sev_group.count(),dask="parallelized")

    ba_bin_name = area_per_sev_bin.name

    #calculate total area burned
    #depending whether using parallelised computing
    try:
        total_area_burned = xr.apply_ufunc(calc_area,ba.count()).values
    except: 
        total_area_burned = xr.apply_ufunc(calc_area,ba.count(),dask="parallelized").values
    ## create dataframe with infos
    #area per burn severity
    #dimension name of ndvi_bin object => dim_oder = "ndvi_bins"
    df_area_sev = area_per_sev_bin.to_dataframe(name="area_sev", dim_order=[area_per_sev_bin.dims[0]]).drop(columns = "spatial_ref")
    del area_per_sev_bin
    df_area_sev["area_total"] = total_area_burned
    df_area_sev["percentage_sev"] = (df_area_sev["area_sev"]/df_area_sev["area_total"])*100
    df_area_sev["pixel_sev"] = ba_sev_group.count()
    df_area_sev["pixel_total"] = ba.count().values 
    df_area_sev["percentage_pixel"] = (df_area_sev["pixel_sev"]/df_area_sev["pixel_total"])*100
    if binarized == False:
        #info on burn severity bins
        df_area_sev["interval_mid"] = df_area_sev.index.mid
        df_area_sev["interval_len"] = df_area_sev.index.length

    del ba_sev_group

    ## FWI CLASS
    #total area burned by fwi classes
    #filter fwi by available ba dates
    fwi = fwi.sel(time=ba.time) 
    #classify fwi with np.digitize and increase class values by 1 --> 0-6 => 1-7
    class_fwi = [5.2, 11.2, 21.3, 38, 50, 70] 
    #depending whether using parallelised computing
    try:
        fwi_reclass = xr.apply_ufunc(np.digitize,fwi,class_fwi) + 1
    except:
        #using dask built in fuction
        fwi_reclass = xr.apply_ufunc(da.digitize,fwi,class_fwi,dask="allowed") + 1

    #group burned area by fwi
    if info == True:
        print("grouping burned area by fwi")    
    ba_fwi_group = ba.groupby(fwi_reclass)
    #count burned area by fwi class and calculate area
    try:
        area_per_fwiclass = xr.apply_ufunc(calc_area,ba_fwi_group.count())
    except:
        area_per_fwiclass = xr.apply_ufunc(calc_area,ba_fwi_group.count(),dask="parallelized")
    fwi_bin_name = area_per_fwiclass.name

    #area per fwi class 
    #dimension name of groupby object is name of original fwi variable => dim_order = fwi.name (may change depending on fwi parameter input)
    df_area_fwi = area_per_fwiclass.to_dataframe(name = "area_fwi", dim_order=[fwi.name]).drop(columns = "spatial_ref")
    del area_per_fwiclass
    df_area_fwi["area_total"] = total_area_burned
    df_area_fwi["percentage_fwi"] = (df_area_fwi["area_fwi"]/df_area_fwi["area_total"])*100
    df_area_fwi["pixel_fwi"] = ba_fwi_group.count()
    df_area_fwi["pixel_total"] = ba.count().values  #fwi.where(ba,drop=True).count().values #fwi.count().values #..> fwi.count() counts
    df_area_fwi["percentage_pixel"] = (df_area_fwi["pixel_fwi"]/df_area_fwi["pixel_total"])*100

    #round floats
    df_area_sev = df_area_sev.round(2)
    df_area_fwi = df_area_fwi.round(2)
    
    return df_area_sev, df_area_fwi

def lin_reg_firedanger_dict(X_0, y_0, X_name, y_name,path_out,show_plot = False,dict_in=None):
    '''
     X explanatory/independent variable e.g. FWI
     y dependent variable e.g. Burned Area

    linear regression 
    - statistics
    - plotting
    Pearsons correlation coefficent

     version: script dictionary
    '''
    
    #ravel data
    X_0 = X_0.values.ravel()
    y_0 = y_0.values.ravel()
        
    #round decimals (so number not too large for np.float32)
    X_0 = np.around(X_0, decimals = 4)
    y_0 = np.around(y_0, decimals = 4)
    
    #mask data for nan values, both in X and y (though X is more relevant)
    mask = ~np.isnan(X_0) & ~np.isnan(y_0) 
    X = X_0[mask]
    y = y_0[mask]
    
    #split data in in train / test data
    #reshape X -> n samples, 1 feature
    X_train, X_test, y_train, y_test_0 = train_test_split(X.reshape(-1,1), y, test_size=0.3, random_state=42)

    #linear regression model object
    lr = linear_model.LinearRegression()
    #train model
    lr.fit(X_train, y_train)
    #make predictions
    y_predict = lr.predict(X_test)
    
    #reshape data
    y_test = y_test_0.reshape(-1, 1)
    y_predict = y_predict.reshape(-1,1)
    
    #pearsons correlation coefficient
    pr = r_regression(X.reshape(-1,1),y.ravel())
    
    lin_reg_dict = {
        "X" : X_name,
        "y" : y_name,
        "n_samples" : len(X_0),
        "n_samples_masked" : len(X),
        "X_unique" : len(np.unique(X)),
        "y_unique" : len(np.unique(y)),
        "ratio_unique_y_X" : (len(np.unique(y)) / len(np.unique(X))),
        "X_train_shape" : X_train.shape,
        "y_train_shape" : y_train.shape,
        "X_test_shape" : X_test.shape,
        "y_test_shape" : y_test_0.shape,
        f"coeff_{X_name}_{y_name}" : round(float(lr.coef_),4),
        f"r2_{X_name}_{y_name}" : round(float(r2_score(y_test, y_predict)),4),
        f"rmse_{X_name}_{y_name}" : round(float(mean_squared_error(y_test, y_predict)),4),
        f"pearson_{X_name}_{y_name}" : round(float(pr),4),
        f"mape_{X_name}_{y_name}" : round(float(mean_absolute_percentage_error(y_test, y_predict)),4)

    }

    #plot data
    #plot figure
    fig,ax=plt.subplots()
    fig.subplots_adjust(top=0.85)

    # fig.suptitle(f"Linear correlation between {X_name} (explanatory variable) and {y_name} (dependent variable)", fontsize=12)
    fig.suptitle(f"Linear correlation between {X_name} and {y_name} (dependent variable)", fontsize=12)
    ax.set_title(f"Parameter: Effect {dict_in['effect']}, Weight 1/{dict_in['weight_p']}, VCI factor {round(float(dict_in['vci_factor_mean']),2)} - Pearson: {round(float(pr),2)}",fontsize=9)
    ax.set_xlabel("Fire Weather Index")
    ax.set_ylabel("Burned Area Burn Severity")
    # ax.text(0.1, 0.9, f"pearson's coefficiens: {round(float(pr),2)}", size=15, color='black')

    ax.scatter(X_test, y_test, color="black")
    ax.plot(X_test, y_predict, color="orange")
        
    plt.xlim([0, 80])
    plt.ylim([-0.5, 0.2])

    # plt.savefig(f"{path_out}\\lincorr_{X_name}_{y_name}_years{'_'.join(map(str,dict_in['years_list']))}_months{'_'.join(map(str,dict_in['months_list']))}_e{dict_in['effect']}_p{dict_in['weight_p']}_nf{str(dict_in['ndvi_factor_active'])}.png")
    plt.savefig(f"{path_out}\\lincorr_{X_name}_{y_name}_years{dict_in['years_list']}_months{dict_in['months_list']}_e{dict_in['effect']}_p{dict_in['weight_p']}.png") #_nf{str(dict_in['ndvi_factor_active'])}.png")

    if show_plot == True:
        #show figure
        plt.show()
    
    plt.close()
            
    
    return lin_reg_dict

def lin_reg_pearson_dict(X_0, y_0, X_name, y_name):
    '''
    X explanatory/independent variable e.g. VI
    y dependent variable e.g. FWI
    
    return: dict containing
        - name of variables (str)
        - pearson coefficient (float)
        - f statistic for each feature (float) 
        - p values associated with the f statistic (float)
    '''
    #ravel data
    X_0 = X_0.values.ravel()
    y_0 = y_0.values.ravel()
        
    #round decimals (so number not too large for np.float32)
    X_0 = np.around(X_0, decimals = 4)
    y_0 = np.around(y_0, decimals = 4)
    
    #mask data for nan values, both in X and y (though X is more relevant)
    mask = ~np.isnan(X_0) & ~np.isnan(y_0) 
    X = X_0[mask]
    y = y_0[mask]
    
    #pearsons correlation coefficient
    pr = r_regression(X.reshape(-1,1),y.ravel())
    round(float(pr),4)

    # fstat_ft,p_values = f_regression(X.reshape(-1,1),y.ravel()) 

    pearson_dict = {
        f"pearson_{X_name}_{y_name}" :round(float(pr),2),
        # f"f_stats_feat_{X_name}_{y_name}" : np.round(float(fstat_ft),2),
        # f"p_values_{X_name}_{y_name}" : np.round(float(p_values),2)
    }
    
    return pearson_dict

def predict_valid_firedanger (fwi, ba_bin, X_name, y_name, path_out, info=True,show_plot = False,dict_in=None):
    '''
    apply sklearn random forest classification with 
        X: classified FWI values (0-7)
        y: binarized burned area data with 0 = true no burned areas, 1 = true burned areas

    input parameters:
    - fwi: (adapted) fwi xarray (xarray dataarray)
    - ba_bin: ba_mask.burned_bin with binarized values (xarray dataarray) (= burned area output from "ba_preprocessing_mask.py")
        - when using burned area output from "ba_preprocessing_composites.py":
            ba_bin: ba_mask.burn_severity with true no burned areas of value 0
    - info (bool):
        if true: prints print statements

    returns: 
    - plots confusion matrix (sklearn.metrics.plot_confusion_matrix)
    - confusion matrix (sklearn.metrics.confusion_matrix)
    '''
    #classify fwi with np.digitize and increase class values by 1 --> 0-6 => 1-7
    class_fwi = [5.2, 11.2, 21.3, 38, 50, 70] 
    #depending whether using parallelised computing
    try:
        fwi_reclass = xr.apply_ufunc(np.digitize,fwi,class_fwi) + 1
    except:
        #using dask built in fuction
        fwi_reclass = xr.apply_ufunc(da.digitize,fwi,class_fwi,dask="allowed") + 1
    
    #calculate stats
    # df_ba, df_fwi = calculate_ba_fwi_stats(fwi,ba)
    df_ba, df_fwi = calculate_ba_fwi_stats(fwi,ba_bin,binarized=True,info=info)
    if info == True:
        print("statistics burned area per severity")
        print(df_ba[["area_sev", "percentage_sev", "pixel_total", "percentage_pixel"]])
        
        print("statistics burned area per severity")
        print(df_fwi[["area_fwi", "percentage_fwi", "pixel_total", "percentage_pixel"]])
    
    #classify data
    X_0 = fwi_reclass.values.ravel()
    y_0 = ba_bin.values.ravel()

    #round decimals (so number not too large for np.float32)
    X = np.around(X_0, decimals = 4)
    y = np.around(y_0, decimals = 4)

    #mask data for nan values, both in X and y (though X is more relevant)
    mask = ~np.isnan(X) & ~np.isnan(y) 
    X = X[mask]
    y = y[mask]

    if info == True:
        print ("number of samples: ", len(X_0))
        print (f"number of samples with applied nan mask: {len(X)}")
        print (f"number of unique X values ({X_name}): {len(np.unique(X))}")
        print (f"number of unique y values ({y_name}): {len(np.unique(y))}")
        print (f"ratio of unique values y/X: %.2f" % (len(np.unique(y)) / len(np.unique(X))))

    #split data in in train / test data
    #reshape X -> n samples, 1 feature
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1), y, test_size=0.3,random_state=42)
    
    #apply random forest classification
    rf = RandomForestClassifier(n_estimators=500)#, oob_score=True)
    rf = rf.fit(X_train, y_train) 
    # rf = rf.fit(X_resampled, y_resampled) 

    y_pred = rf.predict(X_test)

    #reshape data
    y_test = y_test.reshape(-1, 1)
    y_pred = y_pred.reshape(-1,1)

    #calculate metrics
    tpfptnfn = segm.tpfptnfn(y_test, y_pred, None)
    metrics_dict = segm.segmentation_metrics(tpfptnfn)
    
    #plot data
    #confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
    #plot confusion matrix
    fig,ax=plt.subplots()
    fig.subplots_adjust(top=0.85)

    # fig.suptitle(f"Confusion matrix: Burned Area (1) - No Burned Area (0) - {'_'.join(map(str,dict_in['months_list']))}/{'_'.join(map(str,dict_in['years_list']))}", fontsize=12)
    fig.suptitle(f"Confusion matrix: Burned Area (1) - No Burned Area (0) - {dict_in['months_list']}/{dict_in['years_list']}", fontsize=12)
    ax.set_title(f"Parameter: Effect {dict_in['effect']}, Weight 1/{dict_in['weight_p']}, VCI factor {round(float(dict_in['vci_factor_mean']),2)} - Iou {round(float(metrics_dict['iou']),2)}, Accuracy {round(float(metrics_dict['acc']),2)}",fontsize=9) #(include NDVI: {dict_in['ndvi_factor_active']})

    ConfusionMatrixDisplay.from_predictions(y_test,y_pred, cmap="Blues",ax=ax)
    # ConfusionMatrixDisplay.from_estimator(rf,X_test,y_test)

    # plt.savefig(f"{path_out}\\cf_{X_name}_{y_name}_years{'_'.join(map(str,dict_in['years_list']))}_months{'_'.join(map(str,dict_in['months_list']))}_e{dict_in['effect']}_p{dict_in['weight_p']}_nf{str(dict_in['ndvi_factor_active'])}.png")
    plt.savefig(f"{path_out}\\cf_{X_name}_{y_name}_years{dict_in['years_list']}_months{dict_in['months_list']}_e{dict_in['effect']}_p{dict_in['weight_p']}.png")

    if show_plot == True:
        #show figure
        plt.show()
    
    plt.close()
    
    #add confusion matrix to dict
    metrics_dict["firedanger_pred_confmatrix"] = conf_matrix

    #add statistics to metrics
    metrics_dict["df_ba_classified"] = df_ba
    metrics_dict["df_fwi_classified"] = df_fwi
    
    # get iou for each classes
    # seperate class arrays
    y_true_c1 = np.where(y_test == 1, 1, 0)
    y_true_c2 = np.where(y_test == 0, 1, 0)
    y_pred_c1 = np.where(y_pred == 1, 1, 0)
    y_pred_c2 = np.where(y_pred == 0, 1, 0)

    #calculate metrics for each class
    tpfptnfn = segm.tpfptnfn(y_true_c1, y_pred_c1, None)
    c1_metrics = segm.segmentation_metrics(tpfptnfn)
    iou1 = segm._intersection_over_union(tpfptnfn["tp"], tpfptnfn["fp"], tpfptnfn["fn"], tpfptnfn["tn"])
    tpfptnfn = segm.tpfptnfn(y_true_c2, y_pred_c2, None)
    c2_metrics = segm.segmentation_metrics(tpfptnfn)
    iou2 = segm._intersection_over_union(tpfptnfn["tp"], tpfptnfn["fp"], tpfptnfn["fn"], tpfptnfn["tn"])
    miou = (iou1 + iou2) / 2 
    c1_metrics["miou"] = miou
    c2_metrics["miou"] = miou

    metrics_dict["metrics_class_1burned"] = c1_metrics
    metrics_dict["metrics_class_0noburned"] = c2_metrics

    return y_test,y_pred,metrics_dict   


def run_analysis (fwi,ba,ndvi_dif,ndvi, vci, dict_input, path_output = None, fwi_type = "FWI",show_plot_b=False,info_b=False,reproject=False): 
    '''
    input param:
    fwi,ba,ndvidif,ndvi,vci (xarray dataset) (of equal selected time)
    dictionary with input parameter for fwi_plus

    applying the functions:
    - resample all input data to laea 300m
    - apply_ba_mask
    - lin_reg_firedanger_dict
    - calculate_ba_fwi_stats
    - lin_reg_pearson
    - predict_valid_firedanger

    returns: merged dictionary with results of each analysis
    '''
    if reproject == True:
        #reproject data to laea crs with 300m resolution
        fwi = fwi.rio.reproject(dst_crs=3035, resolution=300, nodata=np.nan)
        ba = ba.rio.reproject(dst_crs=3035, resolution=300, nodata=np.nan)
        ndvi_dif = ndvi_dif.rio.reproject(dst_crs=3035, resolution=300, nodata=np.nan)
        ndvi = ndvi.rio.reproject(dst_crs=3035, resolution=300, nodata=np.nan)
        vci = vci.rio.reproject(dst_crs=3035, resolution=300, nodata=np.nan)
    
    #apply burned area mask to ndvi difference
    if info_b==True:
        print("apply burned area mask to ndvi difference")
    fwi_ba_dates, burned_area_pn, burned_area_p,burnedarea_binarized = apply_ba_mask(fwi,ba,ndvi_dif,info_a=info_b)
       
    #linear regression of ba fwi
    if info_b==True:
        print(f"linear regression of burned area and {fwi_type}")
    #linear regression with detected burned areas only
    lin_reg_dict = lin_reg_firedanger_dict(fwi_ba_dates, burned_area_p, fwi_type, "BA",show_plot=show_plot_b,dict_in=dict_input,path_out=path_output)
    #linear regression with burned and sampled no burned areas
    # lin_reg_dict = lin_reg_firedanger_dict(fwi_ba_dates_day, burned_area_pn_, "FWI", "BA",show_plot=show_plot_b,dict_in=dict_input,path_out=path_output)

    stats_dict = dict_input | lin_reg_dict
        
    #calculate ba fwi stats
    if info_b==True:
        print(f"calculate area affected based on burned area and {fwi_type}")
    df_area_sev, df_area_fwi = calculate_ba_fwi_stats(fwi_ba_dates,burned_area_p,info=info_b)
    stats_dict["df_area_sev"] = df_area_sev
    stats_dict["df_area_fwi"] = df_area_fwi
    
    #korrelation ndvi fwi
    if info_b==True:
        print("compute pearson coefficient of fwi and ndvi/vci")
    pearson_dict = lin_reg_pearson_dict(ndvi,fwi, "NDVI", fwi_type)
    stats_dict_pear_ndvi = stats_dict | pearson_dict 
    
    #korrelation vci fwi
    pearson_dict = lin_reg_pearson_dict(vci,fwi, "VCI", fwi_type)
    stats_dict_pear_vci = stats_dict_pear_ndvi | pearson_dict

    #classification and prediction of fire danger on burned/no burned areas
    if info_b==True:
        print("classify and predict fire danger on burned/no burned areas (random forest)")
    # y_predict, y_test, metrics = predict_valid_firedanger(fwi_ba_dates,ba.burned_bin,"FWI", "binarized Burned Areas",info=info_b,show_plot=show_plot_b,dict_in=dict_input,path_out=path_output)
    y_predict, y_test, metrics = predict_valid_firedanger(fwi_ba_dates,burnedarea_binarized,fwi_type, "binarized Burned Areas",info=info_b,show_plot=show_plot_b,dict_in=dict_input,path_out=path_output)
    #merge dictionaries
    dict_merge = stats_dict_pear_vci | metrics
    
    return dict_merge

