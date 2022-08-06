#plot daily vci
import xarray as xr
import os
from os import listdir
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
import imageio
import numpy as np
import aux_functions

#turn interactive plotting off
matplotlib.use('Agg')

path_data = "D:\\roll_thesis\\data_fwi_vi"

#path to merged daily vci data per year
path_input = f"{path_data}\\output\\vci_preproc"

#set color ramp of vci plot
levels_vci = list(np.arange(0,1,0.005))
colors_vci = "RdYlGn"

##loop through available yearly data
for file in listdir(path_input):
    if file.endswith(".nc"):
        area = file.split("_")[1]
        year = file.split("_")[2].split(".")[0]
 
        path = join(path_input, file)
        vci_year = xr.load_dataset(path).rio.write_crs(4326)
        # print(vci_year)

        #output folder for year
        plot_output = f"{path_input}\\plot\\{year}"
        if not os.path.exists(plot_output):
            os.makedirs(plot_output)

        #loop over each day of year
        print(f"plotting year {year}")
        #collecting filenames for creating a gif per year
        fig_list = []

        #counter for progress bar
        counter = 0
        total_files = len(vci_year.time) 

        for day in vci_year.time:
            plot_day = str(day.values)[:10]

            vci_year.sel(time=plot_day).vci.plot(levels=levels_vci, colors=colors_vci, figsize=(11,8))
            plt.suptitle(f"VCI at 12:00 UTC for region {area}", fontsize=16)
            plt.title(f"{plot_day}",fontsize=14, pad=20)
            plt.xlabel("longitude")
            plt.ylabel("latitude")

            fig_out = f"{plot_output}\\vci_{area}_{plot_day}.jpg"

            try:
                plt.savefig(fig_out)
                plt.close()
            except:
                print("file {fig_out} is in use of another program and cannot be overwritten")
                print("day {plot_day} is skipped for plotting")

            #add figures to list for gif
            fig_list.append(imageio.imread(fig_out))

            #print progress
            aux_functions.progress(counter, total_files)
            counter += 1

        #create gif
        imageio.mimsave(f"{path_input}\\vci_{area}_{year}.gif", fig_list)