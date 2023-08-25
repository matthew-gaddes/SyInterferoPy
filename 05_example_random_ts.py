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
from syinterferopy.random_generation import create_random_ts_1_volc


#%% 0: Things to set

srtm_tools_dir = Path('/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-3.0.0')             # SRTM-DEM-tools will be needed for this example.  It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
gshhs_dir = Path("/home/matthew/university_work/data/coastlines/gshhg/shapefile/GSHHS_shp")                    # coastline information, available from: http://www.soest.hawaii.edu/pwessel/gshhg/

SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                      'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                      'SRTM3_tiles_folder'       : Path('./SRTM3/'),                                   # folder for DEM tiles.  
                      'download'                 : True,                                         # If tile is not available locally, try to download it
                      'void_fill'                : True,                                         # some tiles contain voids which can be filled (slow)
                      'gshhs_dir'                : gshhs_dir}                            # srmt-dem-tools needs access to data about coastlines

                           

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



#%% 2: Make time series for that volcano.  

create_random_ts_1_volc(outdir = Path('./05_example_outputs/'), dem_dict = volcano_dems2[0], n_pix = 224, d_start = "20141231", d_stop = "20230801",
                        n_def_location = 2, n_tcs = 2, n_atms = 2,
                        topo_delay_var = 0.00005, turb_aps_mean = 0.02)
    
    
    

