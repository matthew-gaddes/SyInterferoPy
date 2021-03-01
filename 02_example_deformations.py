#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:53:45 2020

@author: matthew
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append('./lib/')
from syinterferopy_functions import deformation_wrapper
from auxiliary_functions import griddata_plot                        # smaller functions


#%% Things to set
sys.path.append(str(Path('/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/')))                         # SRTM-DEM-tools will be needed for this example.  
                                                                                                                 # It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
from dem_tools_lib import SRTM_dem_make


## Campi Flegrei
dem_loc_size = {'centre'        : (14.14, 40.84),
                'side_length'   : (20e3,20e3)}                                   # lon lat width height (m) of interferogram.  
deformation_ll = (14.14, 40.84,)                                                     # deformation lon lat
mogi_kwargs = {'volume_change' : 1e6,                           
                'depth'         : 2000}                                                 # both in metres

dem_settings = {"download"              : False,                                 # if don't need to download anymore, faster to set to false
                "void_fill"             : False,                                 # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "SRTM3_tiles_folder"    : './SRTM3/',                            # folder to keep SRTM3 tiles in 
                "water_mask_resolution" : 'f'}                                   # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)

#%% Login details are now needed to download SRTM3 tiles:
    
ed_username = input(f'Please enter your USGS Earthdata username:  ')
ed_password = input(f'Please enter your USGS Earthdata password (NB characters will be visible!   ):  ')

dem_settings['ed_username'] = ed_username                                                                   # append to the dict of dem_settings so it can be passed to SRTM_dem_make quickly.  
dem_settings['ed_password'] = ed_password


#%%  First, make a DEM (digtal elevation model)

dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg, "01 A digital elevation model (DEM) of Campi Flegrei.  ")

ll_extent = [(lons_mg[-1,0], lats_mg[-1,0]), (lons_mg[1,-1], lats_mg[1,-1])]                # [lon lat tuple of lower left corner, lon lat tuple of upper right corner]

#%% With a dem, we can now generate geocoded deformation patterns.  

# 1: Mogi
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'mogi', dem, **mogi_kwargs)
griddata_plot(los_grid, lons_mg, lats_mg, '02: Point (Mogi) source', dem_mode = False)



# 2: Dyke
dyke_kwargs = {'strike' : 0,                                    # in degrees
               'top_depth' : 1000,                              # in metres
               'bottom_depth' : 3000,
               'length' : 5000,
               'dip' : 80,                                      # in degrees
               'opening' : 0.5}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'dyke', dem, **dyke_kwargs)
griddata_plot(los_grid, lons_mg, lats_mg, '03: Opening Dyke', dem_mode = False)


#3: Sill
sill_kwargs = {'strike' : 0,                            # degrees
               'depth' : 3000,                          # metres
               'width' : 5000,
               'length' : 5000,
               'dip' : 1,                               # degrees
               'opening' : 0.5}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'sill', dem, **sill_kwargs)
griddata_plot(los_grid, lons_mg, lats_mg, '04: Inflating Sill', dem_mode = False)


#4: SS EQ
quake_ss_kwargs  = {'strike' : 0,                       # degrees
                    'dip' : 80,
                    'length' : 5000,                        # metres
                    'rake' : 0,
                    'slip' : 1,
                    'top_depth' : 4000,
                    'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'quake', dem, **quake_ss_kwargs)
griddata_plot(los_grid, lons_mg, lats_mg, '05: SS fault earthquake - is magnitude correct?', dem_mode = False)

#5: Normal EQ
quake_normal_kwargs  = {'strike' : 0,
                        'dip' : 70,
                        'length' : 5000,
                        'rake' : -90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'quake', dem, **quake_normal_kwargs)
griddata_plot(los_grid, lons_mg, lats_mg, '06: Normal fault earthquake - is magnitude correct?', dem_mode = False)


#6: Thrust EQ
quake_thrust_kwargs = {'strike' : 0,
                       'dip' : 30,
                       'length' : 5000,
                       'rake' : 90,
                       'slip' : 1,
                       'top_depth' : 4000,
                       'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'quake', dem, **quake_thrust_kwargs)
griddata_plot(los_grid, lons_mg, lats_mg, '07: Thurst fault earthquake - is magnitude correct?', dem_mode = False)        
        
        

        

        