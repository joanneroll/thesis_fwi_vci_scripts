# firemaster_scripts

masterthesis 
"Remote sensing-based adaptation of the Canadian Fire Weather Index towards developing a new fire danger monitoring system for Europe"

**Anaconda Version**
- 4.13.0

**Python Version**
- Python 3.9.7

*important Packages*
- xarray (2022.3.0)
- netcdf4 (1.5.8)
- h5netcdf (1.5.7)
- cfgrib (0.9.10.1)
- rioxarray (0.9.0)
- geopandas (0.10.0)
- shapely (1.8.0)
- pandas (1.4.2)
- numpy (1.20.3)
- geocube (0.2.0)
- scikit-learn (1.0.1)
- ukis-metrics (0.1.3)
- numba (0.54.1)

**order of scripts**
- prep_input_param.py

- clip_stack_era5_indices.py
- mask_resample.py

- fwi_main.py
- fwi_wang.py

- ndvi_preprocessing.py
- ndvi_smooth_difference.py
- calc_vci.py

- ba_preprocessing_mask.py

- analysis_preprocessing.py
- test_param_newfwi.py

*function collections*
- aux_functions.py
- analysis_functions.py

*plotting of vegetation indices*
- plot_ndvi.py
- plot_vci.py
