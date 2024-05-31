#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:11:06 2024

@author: matthew
"""

import numpy as np
import numpy.ma as ma
import sys
import time
import pdb
from pathlib import Path
import matplotlib.pyplot as plt


import syinterferopy
# from syinterferopy.syinterferopy import atmosphere_topo, atmosphere_turb, deformation_wrapper, coherence_mask
from syinterferopy.aux import griddata_plot # col_to_ma, plot_ifgs                        
from syinterferopy.comsol import sypy_dem_to_comsol_dem




#%% ################################ Things to set ################################
np.random.seed(0)

srtm_tools_dir = Path('/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-3.1.0')             # SRTM-DEM-tools will be needed for this example.  It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
gshhs_dir = Path("/home/matthew/university_work/data/coastlines/gshhg/shapefile/GSHHS_shp")                    # coastline information, available from: http://www.soest.hawaii.edu/pwessel/gshhg/




dem_settings = {"download"              : False,                                 # if don't need to download anymore, faster to set to false
                "void_fill"             : False,                                 # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "SRTM3_tiles_folder"    : Path('./SRTM3/'),                            # folder to keep SRTM3 tiles in 
                "water_mask_resolution" : 'f',                                   # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)
                'gshhs_dir'             : gshhs_dir}                            # srmt-dem-tools needs access to data about coastlines

n_interferograms = 20                                                            # number of interferograms in the time series

#%% Import srtm_dem_tools

sys.path.append(str(srtm_tools_dir))                         # 
import srtm_dem_tools
from srtm_dem_tools.constructing import SRTM_dem_make

#%% Login details are now needed to download SRTM3 tiles:
    
# ed_username = input(f'Please enter your USGS Earthdata username:  ')
# ed_password = input(f'Please enter your USGS Earthdata password (NB characters will be visible!   ):  ')

# dem_settings['ed_username'] = ed_username
# dem_settings['ed_password'] = ed_password

#%%  Campi Flegrei

## Campi Flegrei
dem_loc_size = {'centre'        : (14.14, 40.84),
                'side_length'   : (20e3,20e3)}                                   # lon lat width height (m) of interferogram.  

dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg, "01 A digital elevation model (DEM) of Campi Flegrei.  ")
sypy_dem_to_comsol_dem(dem, lons_mg, lats_mg, outfile = Path("./comsol_dem_campi_flegrei.txt"))

#%%  Vesuvius

dem_loc_size = {'centre'        : (14.43, 40.82),
                'side_length'   : (20e3,20e3)}                                   # lon lat width height (m) of interferogram.  

dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg, "02 A digital elevation model (DEM) of Vesuvius.  ")
sypy_dem_to_comsol_dem(dem, lons_mg, lats_mg, outfile = Path("./comsol_dem_vesuvius.txt"))
