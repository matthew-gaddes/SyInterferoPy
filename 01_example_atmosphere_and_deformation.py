#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:02 2019

@author: mgaddes
"""

import numpy as np
import numpy.ma as ma
import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('./lib/')

from syinterferopy_functions import atmosphere_topo, atmosphere_turb, deformation_wrapper, coherence_mask
from auxiliary_functions import col_to_ma, griddata_plot, plot_ifgs                        



#%% ################################ Things to set ################################
np.random.seed(0)

sys.path.append(str(Path('/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/')))                         # SRTM-DEM-tools will be needed for this example.  It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
from dem_tools_lib import SRTM_dem_make

## Campi Flegrei
dem_loc_size = {'centre'        : (14.14, 40.84),
                'side_length'   : (20e3,20e3)}                                   # lon lat width height (m) of interferogram.  
deformation_ll = (14.14, 40.84,)                                                 # deformation lon lat
mogi_kwargs = {'volume_change' : 1e6,                           
                'depth'         : 2000}                                          # both in metres

dem_settings = {"download"              : False,                                 # if don't need to download anymore, faster to set to false
                "void_fill"             : False,                                 # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "SRTM3_tiles_folder"    : './SRTM3/',                            # folder to keep SRTM3 tiles in 
                "water_mask_resolution" : 'f'}                                   # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)

n_interferograms = 20                                                            # number of interferograms in the time series

#%%  First, let's look at making DEMS (digtal elevation models)

dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg, "01 A digital elevation model (DEM) of Campi Flegrei.  ")


#lets change "scene_centre" to look somewhere else
dem_loc_centre = {'centre' : (14.434, 40.816),'side_length' : (20e3, 20e3)}                                 # lon lat, width, height (m), note small change to lon lat
dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg,  "02 DEM for new location (Vesuvius)")

# Or we can change the 'width' argument of "scene_centre" to a bigger value, and bigger scene.  
dem_loc_centre = {'centre' : (14.43, 40.82), 'side_length' :  (40e3, 40e3)}                                 # lon lat scene width(km), note change to 40
dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg,  "03 DEM for new location (Vesuvius), 40km scene")

# and reset it to Campi Flegrei (and make a good, void filled one, which will be slow)
dem_settings['void_fill'] = True                                                                            # turn void filling on


dem_loc_size = {'centre'        : (14.14, 40.84), 'side_length'   : (20e3,20e3)}                # lon lat width height (m) of interferogram.  
dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
griddata_plot(dem, lons_mg, lats_mg,  "05 back to the original, 20km scene")

water_mask = ma.getmask(dem)                                                                    # DEM has no values for water, and we have no radar return from water, so good to keep a mask available



#%% Second, make a deformation signal

signals_m = {}                                                                              # these should be in metres

ll_extent = [(lons_mg[-1,0], lats_mg[-1,0]), (lons_mg[1,-1], lats_mg[1,-1])]                # [lon lat tuple of lower left corner, lon lat tuple of upper right corner]

los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem, lons_mg, lats_mg, deformation_ll, source = 'mogi', **mogi_kwargs)
signals_m["deformation"] = ma.array(los_grid, mask = water_mask)
griddata_plot(signals_m["deformation"], lons_mg, lats_mg, "06 Deformaiton signal", dem_mode = False)                         # plot

# we can offset of the deformaiton signal (ie set a new lon lat)
deformation_ll = (14.17, 40.88,)                                                                                                            # deformation lon lat are changed
signals_m["deformation"], x_grid, y_grid, z_grid = deformation_wrapper(dem, lons_mg, lats_mg, deformation_ll, source = 'mogi', **mogi_kwargs)
griddata_plot(signals_m["deformation"], lons_mg, lats_mg, "07 Deformaiton signal - shifted", dem_mode = False)                         # plot


#%% make a topograhically correlated atmospheric phase screen (APS), using the DEM

signals_m['topo_correlated_APS'] = atmosphere_topo(dem, strength_mean = 56.0, strength_var = 12.0, difference = True)                    # dem must be in metres
griddata_plot(signals_m["topo_correlated_APS"], lons_mg, lats_mg, "Topographically correlated APS", dem_mode = False)                         # plot

#%% Make a turbulent APS, correlated on a length scale of 5000m
Lc = 5000
ph_turb = atmosphere_turb(1, lons_mg, lats_mg, water_mask, Lc = Lc, verbose=True, interpolate_threshold = 5e3, mean_m = 0.02)
signals_m["turbulent_APS"] = ph_turb[0,]
griddata_plot(signals_m["turbulent_APS"], lons_mg, lats_mg, f"09 Turbulent APS - correlation length: {Lc}m", dem_mode = False)                         # plot


#%% Combine all the signals to make an interferogram

signals_m["combined"] = ma.zeros((dem.shape))                                               # initate as zeros
for key in signals_m.keys():                                                                # loop through each of the signals made so far
     signals_m["combined"] += signals_m[key]                                                # and add them to the combined signal.  
    
n_pixels = len(ma.compressed(signals_m['deformation']))                                     # get the number of non-masked pixels, consistent across all signals
signals_m_rows = np.zeros((4, n_pixels))                                                    # initiate to store signals as row vectors
for signal_n, key in enumerate(signals_m.keys()):                                           # loop through each signal (defo, topo APS, turb APS, combined)
    signals_m_rows[signal_n,:] = ma.compressed(signals_m[key])                              # and add as a row vector.  
    
plot_ifgs(signals_m_rows, water_mask, title = 'Deformation / Topo correlated APS / Turbulent APS / Combined', n_rows = 1)
  

#%% Note that we can also synthesise areas of incoherece and use these to update the mask

Lcs = [500, 5000]                                                                                   # sets the length scale of the patches of incoherence, in meters
coherence_masks = {}                                                                                # initate as empty
for Lc in Lcs:
    coherence_masks[Lc] = coherence_mask(lons_mg, lats_mg, Lc=Lc, threshold=0.7)                    # if threshold is 0, all of the pixels are incoherent , and if 1, none are.  


f, axes = plt.subplots(1,2)
f.suptitle('Example of coherence masks')
f.canvas.set_window_title('Example of coherence masks')
axes[0].imshow(coherence_masks[Lcs[0]])
axes[0].set_title(f"Length scale: {Lcs[0]}")
axes[1].imshow(coherence_masks[Lcs[1]])
axes[1].set_title(f"Length scale: {Lcs[1]}")

mask_combined = np.logical_or(water_mask, coherence_masks[Lcs[0]])
ifg_with_incoherence = ma.array(ma.getdata(signals_m["combined"]), mask = mask_combined)
griddata_plot(ifg_with_incoherence, lons_mg, lats_mg, "10 Synthetic interferogram with incoherent regions", dem_mode = False)                         # plot
    

#%% Lets make a time series

S = np.vstack((ma.compressed(signals_m["deformation"]), ma.compressed(signals_m["topo_correlated_APS"])))         # signals will be stored as row vectors 

ph_turb_m  = atmosphere_turb(n_interferograms, lons_mg, lats_mg, water_mask, Lc = 2000, verbose=True, interpolate_threshold = 100, mean_m = 0.02)
N = np.zeros((n_interferograms, S.shape[1]))
for row_n, ph_turb in enumerate(ph_turb_m):                                         # conver the noise (turbulent atmosphere) into a matrix of row vectors.  
    N[row_n,] = ma.compressed(ph_turb)

A = np.random.randn(n_interferograms,2)                                             # these column vectors control the strength of each source through time
X = A@S + N                                                                         # do the mixing: X = AS + N
plot_ifgs(X, water_mask, title = 'Synthetic Interferograms')    
    
A[:,0] *= 3                                                                         # increase the strength of the first column, which controls the deformation
X = A@S + N                                                                         # do the mixing: X = AS + N
plot_ifgs(X, water_mask, title = 'Synthetic Interferograms: increased deformation strength')



