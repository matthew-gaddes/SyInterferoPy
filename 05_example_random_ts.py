#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:58:34 2023

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:55:59 2020

"""
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
import pdb

import syinterferopy
from syinterferopy.random_generation import create_random_synthetic_ifgs

#%% Debug functions


def matrix_show(matrix, title=None, ax=None, fig=None, save_path = None, vmin0 = False):
    """Visualise a matrix
    Inputs:
        matrix | r2 array or masked array
        title | string
        ax | matplotlib axes
        save_path | string or None | if a string, save as a .png in this location.  
        vmin0 | boolean | 
        

    2017/10/18 | update so can be passed an axes and plotted in an existing figure
    2017/11/13 | fix bug in how colorbars are plotted.
    2017/12/01 | fix bug if fig is not None
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    matrix = np.atleast_2d(matrix)                   # make at least 2d so can plot column/row vectors

    if isinstance(matrix[0,0], np.bool_):           # boolean arrays will plot, but mess up the colourbar
        matrix = matrix.astype(int)                 # so convert

    if vmin0:
        matrixPlt = ax.imshow(matrix,interpolation='none', aspect='auto', vmin = 0)
    else:
        matrixPlt = ax.imshow(matrix,interpolation='none', aspect='auto')
    fig.colorbar(matrixPlt,ax=ax)
    if title is not None:
        ax.set_title(title)
        fig.canvas.set_window_title(f"{title}")

    if save_path is not None:                                                   # possibly save the figure
        if title is None:                                                       # if not title is supplied, save with a default name
            fig.savefig(f"{save_path}/matrix_show_output.png")
        else:
            fig.savefig(f"{save_path}/{title}.png")                             # or with the title, if it's supplied 
            
    
    plt.pause(1)                                                                    # to force it to be shown when usig ipdb



#%% 0: Things to set

srtm_tools_dir = Path('/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-3.0.0')             # SRTM-DEM-tools will be needed for this example.  It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
gshhs_dir = Path("/home/matthew/university_work/data/coastlines/gshhg/shapefile/GSHHS_shp")                    # coastline information, available from: http://www.soest.hawaii.edu/pwessel/gshhg/

SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                      'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                      'SRTM3_tiles_folder'       : Path('./SRTM3/'),                                   # folder for DEM tiles.  
                      'download'                 : True,                                         # If tile is not available locally, try to download it
                      'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                      'gshhs_dir'                : gshhs_dir}                            # srmt-dem-tools needs access to data about coastlines

# synthetic_ifgs_settings = {'defo_sources'           :  ['no_def', 'dyke', 'sill', 'mogi'],      # deformation patterns that will be included in the dataset.  
#                            'n_ifgs'                 : 7,                                        # the number of synthetic interferograms to generate
#                            'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
#                            'outputs'                : ['uuu', 'uud', 'www', 'wwd', 'rid'],     # channel outputs.  uuu = unwrapped across all 3, uu
#                            'intermediate_figure'    : False,                                     # if True, a figure showing the steps taken during creation of each ifg is displayed.  
#                            'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
#                            'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
#                            'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
#                            'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
#                            'turb_aps_mean'          : 0.02,                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
#                            'turb_aps_length'        : 5000}                                     # turbulent APS will be correlated on this length scale, in metres.  
                           

volcano_dems = [ {'name': 'Campi Flegrei',           'centre': (14.139, 40.827),     'side_length' : (40e3, 40e3)}]                  # centre is lon then lat, side length is x then y, in km.  
                # {'name': 'Witori',                  'centre': (150.516, -5.576),    'side_length' : (40e3, 40e3)},
                # {'name': 'Lolobau',                 'centre': (151.158, -4.92),     'side_length' : (40e3, 40e3)},
                # {'name': 'Krakatau',                'centre': (105.423, -6.102),    'side_length' : (40e3, 40e3)},
                # {'name': 'Batur',                   'centre': (115.375, -8.242),    'side_length' : (40e3, 40e3)},
                # {'name': 'Taal',                    'centre': (120.993, 14.002),    'side_length' : (40e3, 40e3)},
                # {'name': 'Aira',                    'centre': (130.657, 31.593),    'side_length' : (40e3, 40e3)},
                # {'name': 'Asosan',                  'centre': (131.104, 32.884),    'side_length' : (40e3, 40e3)},
                # {'name': 'Naruko',                  'centre': (140.734, 38.729),    'side_length' : (40e3, 40e3)},
                # {'name': 'Towada',                  'centre': (140.88, 40.51),      'side_length' : (40e3, 40e3)}]
                


#%% Import srtm_dem_tools

sys.path.append(str(srtm_tools_dir))                         # 
import srtm_dem_tools
from srtm_dem_tools.constructing import SRTM_dem_make_batch

#%% 1: Create a list of locations (in this case subaerial volcanoes) to make interferograms for, and make them.  

np.random.seed(0)                                                                                           # 0 used in the example

try:
    print('Trying to open a .pkl of the DEMs... ', end = '')
    with open('example_05_dems.pkl', 'rb') as f:
        volcano_dems2 = pickle.load(f)                                                              # keys are ['name', 'centre', 'side_length', 'dem', 'lons_mg', 'lats_mg'])
    f.close()
    print('Done.  ')

except:
    print('Failed.  Generating them from scratch, which can be slow.  ')
    ed_username = input(f'Please enter your USGS Earthdata username:  ')                                        # needed to download SRTM3 tiles
    ed_password = input(f'Please enter your USGS Earthdata password (NB characters will be visible!   ):  ')
    SRTM_dem_settings['ed_username'] = ed_username                                                                   # append to the dict of dem_settings so it can be passed to SRTM_dem_make quickly.  
    SRTM_dem_settings['ed_password'] = ed_password
    volcano_dems2 = SRTM_dem_make_batch(volcano_dems, **SRTM_dem_settings)                                  # make the DEMS, keys are: ['name', 'centre', 'side_length', 'dem', 'lons_mg', 'lats_mg'])
    with open(f'example_05_dems.pkl', 'wb') as f:
        pickle.dump(volcano_dems2, f)
    print('Saved the dems as a .pkl for future use.  ')



#%%

# one function to call on each volcano
# dem should be large. 
# trasnalte def and dem.  
# random time series (x1000)
#  

def create_random_ts_1_volc(dem_dict, n_pix = 224, d_start = "20141231", d_stop = "20230801",
                            n_def_location = 2, n_tcs = 5, n_atms = 5,
                            topo_aps_mean = 56.0, topo_aps_var = 2.0, turb_aps_mean = 0.02):
    """Create random time series of Sentinel-1 data for a given volcano.  
    n_def_location random crops of the DEM with the deformation placed somewhere randomly are made (the majority of the deformation must be on land.  )
    n_tcs patterns for the temporal evolution of the deformation are made.  
    n_atms instances of atmospheric noise are made (turbulent and topographically correlated)
    Total number of time series = n_def_location * n_tcs * n_atms.  
    
    Inputs:
        dem_dict | dict | DEM and vairous parameters.  
        n_pix | int | size out interferograms outp.
        d_start | string | YYYYMMDD of time series start
        d_stop | string | YYYYMMDD of time series end
        
        n_def_locations | int | Number of random DEM crops with random deformation placement.  
        n_tcs | int | Number of random deformation time courses.  
        n_atms | int | Number of random time series atmospheres.  
        
        topo_aps_mean | float | rad/km of delay for the topographically correlated APS
        topo_aps_var | float | rad/km.  Sets the strength difference between topographically correlated APSs
        turb_aps_mean | float | mean strength of turbulent atmospheres, in metres.  Note the the atmosphere_turb funtion takes cmm, and the value provided in m is converted first
        
    Returns:
        Saves pickle file of the time series, the deformaiton time course, and hte DEM.  
        
    History:
        2023_08_24 | MEG | Written.  
    
    """
    

    from syinterferopy.random_generation import generate_dems_and_defos
    from syinterferopy.atmosphere import atmosphere_turb, atmosphere_topos
    from syinterferopy.temporal import defo_to_ts, generate_random_temporal_baselines, sample_deformation_on_acq_dates
    from syinterferopy.temporal import generate_uniform_temporal_baselines, generate_random_tcs
   
    out_file_n = 0
    
    defos, dems = generate_dems_and_defos(dem_dict, n_pix, min_deformation = 0.05, max_deformation = 0.25, n_def_location = n_def_location)             # generate multiple crops of DEM with deformation places randomly on it.  
    tcs, def_dates = generate_random_tcs(n_tcs, d_start, d_stop)                                                                                        # generate random time courses for the deformation. 
    
            
    for defo_n, defo in enumerate(defos):
        for tc_n, tc in enumerate(tcs):
            for atm_n in range(n_atms):
                #acq_dates, tbaselines = generate_random_temporal_baselines(d_start, d_stop)
                acq_dates, tbaselines = generate_uniform_temporal_baselines(d_start, d_stop)
                tc_resampled = sample_deformation_on_acq_dates(acq_dates, tc, def_dates)                                                # get the deformation on only the days when there's a satellite acquisitions.  Note that this is 0 on the first date.  
                defo_ts = defo_to_ts(defo, tc_resampled)
                
                atm_turbs = atmosphere_turb(len(acq_dates)-1, dem_dict['lons_mg'][:n_pix,:n_pix], dem_dict['lats_mg'][:n_pix,:n_pix], mean_m = turb_aps_mean)           # generate some random atmos
                atm_topos = atmosphere_topos(len(acq_dates)-1, dems[defo_n,], topo_aps_mean, topo_aps_var)
                
                ts = defo_ts + atm_turbs + atm_topos
                if atm_n == 0:
                    tss = ma.zeros((n_atms,  ts.shape[0], ts.shape[1], ts.shape[2]))                                                                # initiliase
                tss[atm_n,] = ts
                print(f"Deformation location: {defo_n} Time course: {tc_n} Atmosphere {atm_n}")
                
                
            with open(f'file_{out_file_n:05d}.pkl', 'wb') as f:
                pickle.dump(tss, f)
                pickle.dump(tc, f)
                pickle.dump(dems[defo_n], f)
            out_file_n += 1
                    
    


create_random_ts_1_volc(volcano_dems2[0], n_def_location = 10, n_pix = 224)
    
    
    

