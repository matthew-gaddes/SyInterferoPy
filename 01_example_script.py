#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:02 2019

@author: mgaddes
"""
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from auxiliary_functions import dem_wrapper
from auxiliary_functions import signal_deformation                            # this will generate deformation
from auxiliary_functions import signal_atmosphere_turb                        # and the turbulent atmospheric signal (APS, spatially correlated noise, like a cloudy sky)
from auxiliary_functions import signal_atmosphere_topo                        # and the topographically corrleated atmospheric signal (APS)

from auxiliary_functions import col_to_ma, matrix_show, quick_dem_plot                        # smaller functions




#%% ################################ Things to set ################################

## Campi Flegrei
scene_centre = [(40.84, 14.14), 20]                                                 # lat lon scene width(km) for area covered by interferogram
deformation_centre = [(40.84, 14.14), 2000 , 1e6]                                            # lat lon depth(m) volume change(m) of point (Mogi) source

dem_settings = {"download_dems" : True,                                             # if don't need to download anymore, faster to set to false
                "void_fill"     : False,                                             # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "path_tiles"    : './SRTM_3_tiles',                                 # folder to keep SRTM3 tiles in 
                "pix_in_m"      : 92.6}                                             # pixels in m.  92.6m for a SRTM3 pixel on equator}        


#%%  First, make a DEM (digtal elevation model)

dem, dem_crop, ijk_m, ll_extent = dem_wrapper(scene_centre, **dem_settings)             # make a dem (both original and cropped ones are returned by this function)

# lets look at the big DEM (the file I sent)
quick_dem_plot(dem, "01 A digital elevation model (DEM) of Italy, white is water, heights are in metres")

# lets look at the small dem
quick_dem_plot(dem_crop, "02 DEM Cropped to area of interest")

# lets change "scene_centre" to look somewhere else
scene_centre = [(40.82, 14.43), 20]                                                     # lat lon scene width(km), note small change to lat and lon
dem, dem_crop, ijk_m, ll_extent = dem_wrapper(scene_centre, **dem_settings)             # make a new dem 
quick_dem_plot(dem_crop, "03 DEM cropped to area of interest for new location (Vesuvius")

# Or we can change "scene_centre" to be a bigger scene
scene_centre = [(40.82, 14.43), 40]                                                     # lat lon scene width(km), note change to 40
dem, dem_crop, ijk_m, ll_extent = dem_wrapper(scene_centre, **dem_settings)             # make a new dem 
quick_dem_plot(dem_crop, "04 DEM cropped to area of interest for new location (Vesuvius), 40km scene")

# and reset it to Campi Flegrei (and make a good, void filled one, which will be slow)
dem_settings['void_fill'] = True                                                        # turn void filling on
scene_centre = [(40.84, 14.14), 20]                                                     # lat lon scene width(km)
dem, dem_crop, ijk_m, ll_extent = dem_wrapper(scene_centre, **dem_settings)             # make a new dem 
quick_dem_plot(dem_crop, "05 back to the original, 20km scene")


water_mask = ma.getmask(dem_crop)                                                      # DEM has no values for water, and we have no radar return from water, so good to keep a mask available





#%% Second, make a deformation signal

signals_m = {}                                                                              # these should be in metres

_, signals_m["deformation"] = signal_deformation(dem, water_mask, deformation_centre, scene_centre, ijk_m, ll_extent)
matrix_show(signals_m["deformation"], title = "06 Deformaiton signal")

# we can offset of the deformaiton signal (ie set a new lat lon)
deformation_centre = [(40.83, 14.12), 2000 , 1e6]                                     # lat lon depth(m) volume change(m) of hte deformation signal
_, signals_m["deformation"] = signal_deformation(dem, water_mask, deformation_centre, scene_centre, ijk_m, ll_extent)
matrix_show(signals_m["deformation"], title = "07 Deformaiton signal - shifted")



#%% make a topograhically correlated atmospheric phase screen (APS), using the DEM

signals_m['topo_correlated_APS'] = signal_atmosphere_topo(dem_crop, strength_mean = 56.0, strength_var = 12.0, difference = True)                    # dem must be in metres
matrix_show(signals_m["topo_correlated_APS"], title = "08 Topographically correlated APS")

#%% make a turbulent APS (just spatially correlated noise)

ph_turb, _ = signal_atmosphere_turb(1, water_mask, dem_crop.shape[0], Lc = None, verbose=True, interpolate_threshold = 100, mean_cm = 2)
signals_m["turbulent_APS"] = ph_turb[0,]

matrix_show(signals_m["turbulent_APS"], title = "09 Turbulent APS - just spatially correlated noise")
del ph_turb

#%% Combine all the signals to make an interferogram

signals_m["combined"] = ma.zeros((dem_crop.shape))

for key in signals_m.keys():    
     signals_m["combined"] += signals_m[key]
    
    
# with a figure output
fig1, axes = plt.subplots(1,4)
for plot_n, key in enumerate(signals_m.keys()):    
    matrix_show(signals_m[key], ax = axes[plot_n], fig = fig1, title = key)
    axes[plot_n].set_aspect('equal')


fig1.suptitle('Signals (in metres).  Note that if the turbulent APS is strong, (ie the noise), it might dwarf the other signals')


#%% Lets make a time series


S = np.vstack((ma.compressed(deformation_m), ma.compressed(topo_correlated_APS_m)))                                # signals will be stored as row vectors 

ph_turb_m, _ = signal_atmosphere_turb(n_interferograms, water_mask, dem_crop.shape[0], Lc = None, verbose=True, interpolate_threshold = 100, mean_cm = 2)
N = np.zeros((n_interferograms, S.shape[1]))
for row_n, ph_turb in enumerate(ph_turb_m):
    N[row_n,] = ma.compressed(ph_turb)


A = np.random.randn(n_interferograms,2)                                             # these column vectors control the strength of each source through time

X = A@S + N                                                                         # do the mixing: X = AS + N

# The interferograms are now rows in X
# and we can convert them back to masked arrays using this little function ()
f, axes = plt.subplots(1, n_interferograms)
for counter, row in enumerate(X):
    ifg = col_to_ma(row, pixel_mask = water_mask)
    matrix_show(ifg, title = f"Interferogram {counter}", fig=f, ax = axes[counter])
    axes[counter].set_aspect('equal')
    
    
A[:,0] *= 3                                                                     # first column controls the strength of the deforamation source (A@S, and deformation is first row in S)
                                                                                # so we can make it stronger
X = A@S + N                                                                         # do the mixing: X = AS + N
f, axes = plt.subplots(1, n_interferograms)
for counter, row in enumerate(X):
    ifg = col_to_ma(row, pixel_mask = water_mask)
    matrix_show(ifg, title = f"Interferogram {counter}", fig=f, ax = axes[counter])
    axes[counter].set_aspect('equal')
    
# so now the deformation signal should be a bit clearer in the time series.  
