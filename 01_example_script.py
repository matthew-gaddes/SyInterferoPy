#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:02 2019

@author: mgaddes
"""
import sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from syinterferopy_functions import deformation_Mogi, atmosphere_topo, atmosphere_turb
from auxiliary_functions import dem_wrapper, col_to_ma, matrix_show, quick_dem_plot, plot_ifgs                        # smaller functions



# from dem_tools import dem_wrapper                                             # used to create the DEM.  
# from auxiliary_functions import signal_deformation                            # this will generate deformation
# from auxiliary_functions import signal_atmosphere_turb                        # and the turbulent atmospheric signal (APS, spatially correlated noise, like a cloudy sky)
# from auxiliary_functions import signal_atmosphere_topo                        # and the topographically corrleated atmospheric signal (APS)



#plt.close('all')


#%% ################################ Things to set ################################

## Campi Flegrei
dem_ll_width = [(14.14, 40.84), 20]                                                 # lon lat scene width(km) for area covered by interferogram
deformation_ll = (14.14, 40.84,)                                                     # deformation lon lat
                                                                    #, 2000 , 1e6]                                   # lat lon depth(m) volume change(m) of point (Mogi) source

dem_settings = {"srtm_dem_tools_bin" : '/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/' ,       # The python package SRTM-DEM-tools is required for making DEMs (that are used to synthesise topographically correlated signals).  
                                                                                                            # it can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
                "download_dems"      : False,                                                               # if don't need to download anymore, faster to set to false
                "void_fill"          : False,                                                               # Best to leave as False here, dems can have voids, which it can be worth filling (but can be slow and requires scipy)
                "path_tiles"         : './SRTM_3_tiles/',                                                    # folder to keep SRTM3 tiles in 
                "m_in_pix"           : 92.6,                                                                # pixels in m.  92.6m for a SRTM3 pixel on equator       
                "water_mask_resolution"    : 'i'}                                                                 # resolution of water mask.  c (crude), l (low), i (intermediate), h (high), f (full)

n_interferograms = 20                                                               # number of interferograms in the time series

#%%  First, make a DEM (digtal elevation model)

dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a dem (both original and cropped ones are returned by this function)

# lets look at the big DEM 
quick_dem_plot(dem, "01 A digital elevation model (DEM) of Italy, with no masking of water.  ")

# lets look at the small dem
quick_dem_plot(dem_crop, "02 DEM Cropped to area of interest, with water masked (as white)")

# lets change "scene_centre" to look somewhere else
dem_ll_width = [(14.43, 40.82), 20]                                                                 # lat lon scene width(km), note small change to lat and lon
dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a new dem 
quick_dem_plot(dem_crop, "03 DEM cropped to area of interest for new location (Vesuvius")



# Or we can change the 'width' argument of "scene_centre" to a bigger value, and bigger scene.  
dem_ll_width = [(14.43, 40.82, ), 40]                                                         # lat lon scene width(km), note change to 40
dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(dem_ll_width, **dem_settings)             # make a new dem 
quick_dem_plot(dem_crop, "04 DEM cropped to area of interest for new location (Vesuvius), 40km scene")


# and reset it to Campi Flegrei (and make a good, void filled one, which will be slow)
dem_settings['void_fill'] = True                                                        # turn void filling on
print("quick fix")
scene_centre = [(14.14, 40.84), 20]                                                     # lat lon scene width(km)
dem, dem_crop, ijk_m, ll_extent, ll_extent_crop = dem_wrapper(scene_centre, **dem_settings)             # make a new dem 
quick_dem_plot(dem_crop, "05 back to the original, 20km scene")                         # plot

water_mask = ma.getmask(dem_crop)                                                      # DEM has no values for water, and we have no radar return from water, so good to keep a mask available


import sys; sys.exit()

#%% Second, make a deformation signal

def deformation_wrapper(dem, dem_ll_extent, deformation_ll):
    """ A function to prepare grids of pixels and deformation sources specified in lon lat for use with 
    deformation generating functions that work in metres.  
    
    Inputs:
    Returns:
    History:
    
    """
    
    import numpy as np    
    import numpy.ma as ma
    from auxiliary_functions import ll2xy, crop_matrix_with_ll
    
    deformation_pix = ll2xy((dem_ll_extent[2], dem_ll_extent[0]), 1201,  deformation_ll)            #lat lon of bottom left corner of dem, pixels/deg, and lon lat of deformation centre.  
    
    import ipdb; ipdb.set_trace()
    
    mogi_loc_pix[0,1] = np.size(dem, axis=0) - mogi_loc_pix[0,1]                                                                # convert xy in matrix stlye  (ie from the top left, and not bottom left)
    mogi_loc_m = pix_in_m * mogi_loc_pix                                                                                            # convert to m
    mogi_cent.append(np.array([[mogi_loc_m[0,0], mogi_loc_m[0,1], mogi_cent[1], mogi_cent[2]]]).T)              # (xloc (m), yloc(m), depth (m), volume change (m^3)
    U_mogi = deformation_Mogi(mogi_cent[3],ijk_m,0.25,30e9)                                                                   # 3d displacement
    look = np.array([[np.sin(heading)*np.sin(incidence)],
                      [np.cos(heading)*np.sin(incidence)],
                      [np.cos(incidence)]])                                                                     # ground to satelite unit vector
    U_mogi_los = look.T @ U_mogi                                                                                # 3d displacement converted to LOS displacement
    
    x_pixs = (ll_extent[1] - ll_extent[0])*1200                     # coordinated of points in matrix form (ie 00 is top left)
    y_pixs = (ll_extent[3] - ll_extent[2])*1200
    
   
    defo_signal = np.reshape(U_mogi_los, (y_pixs, x_pixs))                                                        # column vector to rank 2 array


    
    defo_signal_crop, crop_ll_ur = crop_matrix_with_ll(defo_signal, (ll_extent[2], ll_extent[0]), 
                                                       1200, source_cent[0], source_cent[1])

    defo_signal_crop = ma.array(defo_signal_crop, mask = water_mask)
    
    return defo_signal, defo_signal_crop
    
    

deformation_wrapper(dem_crop, ll_extent_crop, deformation_ll)



#%%

def signal_deformation(dem, water_mask, mogi_cent, source_cent, ijk_m, ll_extent, pix_in_m=92.6, heading=192, incidence=32):
    """ A function to make deformation from a mogi source and crop to size.  
    Inputs:
        dem | r2 array | dem for the region of interest.  Used to set size of deforamtion signal array
        water_mask | r2 array  | used to mask the deformation signal
        mogi_cent | list | (lat, lon), depth (m), volume change (m)
        source_cent | list | (lat, lon), scene size (m)
        ijk_m   | r2 array |x locations of pixels in m
        ll_extent | 
        pix_in_m
        headins
        incidence
        
    """
    import numpy as np    
    import numpy.ma as ma
    from auxiliary_functions import ll2xy, crop_matrix_with_ll
    
    
    mogi_loc_pix = ll2xy((ll_extent[2], ll_extent[0]), 1200, np.array([[mogi_cent[0][0], mogi_cent[0][1]]]))            #centre  
    mogi_loc_pix[0,1] = np.size(dem, axis=0) - mogi_loc_pix[0,1]                                                                # convert xy in matrix stlye  (ie from the top left, and not bottom left)
    mogi_loc_m = pix_in_m * mogi_loc_pix                                                                                            # convert to m
    mogi_cent.append(np.array([[mogi_loc_m[0,0], mogi_loc_m[0,1], mogi_cent[1], mogi_cent[2]]]).T)              # (xloc (m), yloc(m), depth (m), volume change (m^3)
    U_mogi = deformation_Mogi(mogi_cent[3],ijk_m,0.25,30e9)                                                                   # 3d displacement
    look = np.array([[np.sin(heading)*np.sin(incidence)],
                      [np.cos(heading)*np.sin(incidence)],
                      [np.cos(incidence)]])                                                                     # ground to satelite unit vector
    U_mogi_los = look.T @ U_mogi                                                                                # 3d displacement converted to LOS displacement
    
    x_pixs = (ll_extent[1] - ll_extent[0])*1200                     # coordinated of points in matrix form (ie 00 is top left)
    y_pixs = (ll_extent[3] - ll_extent[2])*1200
    
    #import ipdb; ipdb.set_trace()
    
    defo_signal = np.reshape(U_mogi_los, (y_pixs, x_pixs))                                                        # column vector to rank 2 array


    
    defo_signal_crop, crop_ll_ur = crop_matrix_with_ll(defo_signal, (ll_extent[2], ll_extent[0]), 
                                                       1200, source_cent[0], source_cent[1])

    defo_signal_crop = ma.array(defo_signal_crop, mask = water_mask)
    
    return defo_signal, defo_signal_crop



signals_m = {}                                                                              # these should be in metres

_, signals_m["deformation"] = signal_deformation(dem, water_mask, deformation_centre, scene_centre, ijk_m, ll_extent)
matrix_show(signals_m["deformation"], title = "06 Deformaiton signal")

# we can offset of the deformaiton signal (ie set a new lat lon)
deformation_centre = [(40.83, 14.12), 2000 , 1e6]                                     # lat lon depth(m) volume change(m) of hte deformation signal
_, signals_m["deformation"] = signal_deformation(dem, water_mask, deformation_centre, scene_centre, ijk_m, ll_extent)
matrix_show(signals_m["deformation"], title = "07 Deformaiton signal - shifted")



#%% make a topograhically correlated atmospheric phase screen (APS), using the DEM

signals_m['topo_correlated_APS'] = atmosphere_topo(dem_crop, strength_mean = 56.0, strength_var = 12.0, difference = True)                    # dem must be in metres
matrix_show(signals_m["topo_correlated_APS"], title = "08 Topographically correlated APS")

#%% make a turbulent APS (just spatially correlated noise)

ph_turb, _ = atmosphere_turb(1, water_mask, dem_crop.shape[0], Lc = None, verbose=True, interpolate_threshold = 100, mean_cm = 2)
signals_m["turbulent_APS"] = ph_turb[0,]

matrix_show(signals_m["turbulent_APS"], title = "09 Turbulent APS - just spatially correlated noise")
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

