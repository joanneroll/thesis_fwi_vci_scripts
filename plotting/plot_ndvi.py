#plot daily ndvi
import xarray as xr
import os
from os import listdir
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
import imageio
import aux_functions

#turn interactive plotting off
matplotlib.use('Agg')

path_data = "D:\\roll_thesis\\data_fwi_vi"

#path to merged daily ndvi data per year
path_input = f"{path_data}\\output\\ndvi_preproc"

apx_out = "mosaik_30"
apx_title = "Mosaik (30 days)"

#set color ramp of ndvi plot
levels_ndvi = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0] #np.arange(-1.0,1.0,0.2) 
colors_ndvi = "PRGn"

##loop through available yearly data
for file in listdir(path_input):
    if file.endswith(".nc"):
        area = file.split("_")[1]
        year = file.split("_")[2].split(".")[0]
 
        path = join(path_input, file)
        ndvi_year = xr.load_dataset(path).rio.write_crs(4326) #include if fail has values check for proceeding

        #output folder for year
        plot_output = f"{path_input}\\plot\\{year}_{apx_out}"
        if not os.path.exists(plot_output):
            os.makedirs(plot_output)

        #loop over each day of year
        print(f"plotting year {year}")
        #collecting filenames for creating a gif per year
        fig_list = []

        #counter for progress bar
        counter = 0
        total_files = len(ndvi_year.time) 

        for day in ndvi_year.time:
            plot_day = str(day.values)[:10]

            ndvi_year.sel(time=plot_day).ndvi.plot(levels=levels_ndvi, colors=colors_ndvi, figsize=(11,8))
            plt.suptitle(f"NDVI at 12:00 UTC ({apx_title}) for region {area}", fontsize=16)
            plt.title(f"{plot_day}",fontsize=14, pad=20)
            plt.xlabel("longitude")
            plt.ylabel("latitude")

            fig_out = f"{plot_output}\\ndvi_{apx_out}_{area}_{plot_day}.jpg"

            plt.savefig(fig_out)
            plt.close()

            #add figures to list for gif
            fig_list.append(imageio.imread(fig_out))

            #print progress
            aux_functions.progress(counter, total_files)
            counter += 1

        #create gif
        imageio.mimsave(f"{path_input}\\ndvi_{area}_{year}_{apx_out}.gif", fig_list)

        #remove files
        #...
        #import shutil
        #shutil.rmtree(plot_output)
