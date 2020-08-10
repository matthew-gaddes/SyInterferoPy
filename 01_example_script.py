#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:02 2019

@author: mgaddes
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from syinterferopy_functions import atmosphere_topo, atmosphere_turb, deformation_wrapper, coherence_mask
from auxiliary_functions import dem_wrapper, col_to_ma, griddata_plot, plot_ifgs                        



#%% ################################ Things to set ################################

## Campi Flegrei
dem_ll_width = [(14.14, 40.84), 20]                                                 # lon lat scene width(km) for area covered by interferogram
deformation_ll = (14.14, 40.84,)                                                     # deformation lon lat
mogi_kwargs = {'volume_change' : 1e6,                           
                'depth'         : 2000}                                                 # both in metres

dem_settings = {"srtm_dem_tools_bin" : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/' ,       # The python package SRTM-DEM-tools is required for making DEMs (that are used to synthesise topographically correlated signals).  
                                                                                                            # it can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
                "download_dems"      : False,                                                               # if don't need to download anymore, faster to set to false
                "void_fill"          : False,                                                               # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "path_tiles"         : './SRTM_3_tiles/',                                                   # folder to keep SRTM3 tiles in 
                "m_in_pix"           : 92.6,                                                                # pixels in m.  92.6m for a SRTM3 pixel on equator       
                "water_mask_resolution"    : 'f'}                                                           # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)

n_interferograms = 20                                                               # number of interferograms in the time series

#%%  First, make a DEM (digtal elevation model)

dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a dem (both original and cropped ones are returned by this function)

# lets look at the big DEM 
griddata_plot(dem, ll_extent, "01 A digital elevation model (DEM) of Italy, with no masking of water.  ")

# lets look at the small dem
griddata_plot(dem_crop, ll_extent_crop, "02 DEM Cropped to area of interest, with water masked (as white)")


# lets change "scene_centre" to look somewhere else
dem_ll_width = [(14.43, 40.82), 20]                                                                 # lat lon scene width(km), note small change to lat and lon
dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a new dem 
griddata_plot(dem_crop, ll_extent_crop,  "03 DEM cropped to area of interest for new location (Vesuvius")



# Or we can change the 'width' argument of "scene_centre" to a bigger value, and bigger scene.  
dem_ll_width = [(14.43, 40.82, ), 40]                                                         # lat lon scene width(km), note change to 40
dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a new dem 
griddata_plot(dem_crop, ll_extent_crop, "04 DEM cropped to area of interest for new location (Vesuvius), 40km scene")


# and reset it to Campi Flegrei (and make a good, void filled one, which will be slow)
dem_settings['void_fill'] = True                                                        # turn void filling on
print("quick fix")
scene_centre = [(14.14, 40.84), 20]                                                     # lat lon scene width(km)
dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(scene_centre, **dem_settings)             # make a new dem 
griddata_plot(dem_crop, ll_extent_crop, "05 back to the original, 20km scene")                         # plot

water_mask = ma.getmask(dem_crop)                                                      # DEM has no values for water, and we have no radar return from water, so good to keep a mask available


#%% Second, make a deformation signal

signals_m = {}                                                                              # these should be in metres

los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'mogi', **mogi_kwargs)
signals_m["deformation"] = ma.array(los_grid, mask = water_mask)
griddata_plot(signals_m["deformation"], ll_extent_crop, "06 Deformaiton signal", dem_mode = False)                         # plot

# we can offset of the deformaiton signal (ie set a new lon lat)
deformation_ll = (14.17, 40.88,)                                                     # deformation lon lat are changed
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll, source = 'mogi', **mogi_kwargs)
signals_m["deformation"] = ma.array(los_grid, mask = water_mask)
griddata_plot(signals_m["deformation"], ll_extent_crop, "07 Deformaiton signal - shifted", dem_mode = False)                         # plot


#%% make a topograhically correlated atmospheric phase screen (APS), using the DEM

signals_m['topo_correlated_APS'] = atmosphere_topo(dem_crop, strength_mean = 56.0, strength_var = 12.0, difference = True)                    # dem must be in metres
griddata_plot(signals_m["topo_correlated_APS"], ll_extent_crop, "Topographically correlated APS", dem_mode = False)                         # plot

#%% make a turbulent APS (just spatially correlated noise)

ph_turb, _ = atmosphere_turb(1, water_mask, dem_crop.shape[0], Lc = None, verbose=True, interpolate_threshold = 100, mean_cm = 2)
signals_m["turbulent_APS"] = ph_turb[0,]
griddata_plot(signals_m["turbulent_APS"], ll_extent_crop, "09 Turbulent APS - just spatially correlated noise", dem_mode = False)                         # plot

del ph_turb

#%% Combine all the signals to make an interferogram

signals_m["combined"] = ma.zeros((dem_crop.shape))

for key in signals_m.keys():    
     signals_m["combined"] += signals_m[key]
    
    
n_pixels = len(ma.compressed(signals_m['deformation']))                                     # get the number of non-masked pixels, consistent across all signals
signals_m_rows = np.zeros((4, n_pixels))                                                    # initiate to store signals as row vectors
for signal_n, key in enumerate(signals_m.keys()):
    signals_m_rows[signal_n,:] = ma.compressed(signals_m[key])
    
plot_ifgs(signals_m_rows, water_mask, title = 'Deformation / Topo correlated APS / Turbulent APS / Combined', n_rows = 1)
  

#%% Note that we can also synthesise areas of incoherece and use these to update the mask

mask_coherence_scale2 = coherence_mask(dem_crop.shape[0], water_mask, scale=2, threshold=0.7)                      # if threshold is 0, all of the pixels are incoherent , and if 1, none are.  
mask_coherence_scale02 = coherence_mask(dem_crop.shape[0], water_mask, scale=20, threshold=0.6)                      # increasing the scale value slightly changes them to single bigger regions.  

        
f, axes = plt.subplots(1,2)
f.suptitle('Example of coherence masks')
axes[0].imshow(mask_coherence_scale2)
axes[1].imshow(mask_coherence_scale02)

mask_combined = np.logical_or(water_mask, mask_coherence_scale2)
ifg_with_incoherence = ma.array(ma.getdata(signals_m["turbulent_APS"]), mask = mask_combined)
griddata_plot(ifg_with_incoherence, ll_extent_crop, "10 Synthetic interferogram with incoherent regions", dem_mode = False)                         # plot
    

#%% Lets make a time series

S = np.vstack((ma.compressed(signals_m["deformation"]), ma.compressed(signals_m["turbulent_APS"])))         # signals will be stored as row vectors 

ph_turb_m, _ = atmosphere_turb(n_interferograms, water_mask, dem_crop.shape[0], Lc = None, verbose=True, interpolate_threshold = 100, mean_cm = 2)
N = np.zeros((n_interferograms, S.shape[1]))
for row_n, ph_turb in enumerate(ph_turb_m):                                         # conver the noise (turbulent atmosphere) into a matrix of row vectors.  
    N[row_n,] = ma.compressed(ph_turb)

A = np.random.randn(n_interferograms,2)                                             # these column vectors control the strength of each source through time
X = A@S + N                                                                         # do the mixing: X = AS + N
plot_ifgs(X, water_mask, title = 'Synthetic Interferograms')    
    
A[:,0] *= 3                                                                         # increase the strength of the first column, which controls the deformation
X = A@S + N                                                                         # do the mixing: X = AS + N
plot_ifgs(X, water_mask, title = 'Synthetic Interferograms: increased deformation strength')

