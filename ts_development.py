#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:02 2019

@author: mgaddes
"""

import numpy as np
import numpy.ma as ma
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pickle


import syinterferopy
from syinterferopy.syinterferopy import atmosphere_topo, atmosphere_turb, deformation_wrapper, coherence_mask
from syinterferopy.aux import col_to_ma, griddata_plot, plot_ifgs                        



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



def quick_linegraph(tcs, title='', zero=False, ax=None, xvals=None):
    """Visualise a timecourse as a line graph, can handle time courses of different lengths
    Inputs:
        tc | list of array| each array is rank 1
        title | str | figure title
        zero | True or False | if True, line drawn at y=0 on figure
        ax | axes object or None | by setting up an axes before, the linegraph can be plotted in a subplot
        xvals | None or rank 1 array | can set the xvalues of the linegraph.  If None, just numbers them 0 onwards (e.g. 0,1,2,3,4....)

    2017/09/05 | written
    2017/10/02 | update so that can be passed an axes object and plotted as part of a subplot
    2020/01/05 | Add debugging feature (plt.pause ipdb workaround), and test with default as on (True)
    2020/01/05 | Add feature to plot multiple line graphs on same axes (that can also be different lengths)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:                                                              # if no axes provided, initate new ones
        f, ax = plt.subplots()

    for tc in tcs:                                                              # loop through each linegraph to plot
        if xvals is None:                                                       # if no xvals provided, make some
            xvals_plot = np.arange(0,np.size(tc))                               # do the making
        else:
            xvals_plot = xvals                                                  # though if they were provided, copr/rename ready for plotting

        ax.plot(xvals_plot, tc, linestyle = '--', marker='', alpha = 0.5)
        ax.scatter(xvals_plot, tc, marker='o', s = 5,)

    if zero:
        ax.axhline(y=0, color='k', alpha=0.4)
    ax.set_title(title)
    plt.pause(1)                                                                    # workaround to show when using ipdb




#%% ################################ Things to set ################################
np.random.seed(0)

srtm_tools_dir = Path('/home/matthew/university_work/15_my_software_releases/SRTM-DEM-tools-3.0.0')             # SRTM-DEM-tools will be needed for this example.  It can be downloaded from https://github.com/matthew-gaddes/SRTM-DEM-tools
gshhs_dir = Path("/home/matthew/university_work/data/coastlines/gshhg/shapefile/GSHHS_shp")                    # coastline information, available from: http://www.soest.hawaii.edu/pwessel/gshhg/

## Campi Flegrei
dem_loc_size = {'centre'        : (14.14, 40.84),
                'side_length'   : (20e3,20e3)}                                   # lon lat width height (m) of interferogram.  
deformation_ll = (14.14, 40.84,)                                                 # deformation lon lat
mogi_kwargs = {'volume_change' : 1e6,                           
                'depth'         : 2000}                                          # both in metres

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


#%%  First, let's look at making DEMS (digtal elevation models)
make = False
if make:
    dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings)
    with open('temp_ts_dem.pkl', 'wb') as f:
        pickle.dump(dem, f)
        pickle.dump(lons_mg, f)
        pickle.dump(lats_mg, f)
else:
    with open('temp_ts_dem.pkl', 'rb') as f:
        dem = pickle.load(f)   
        lons_mg = pickle.load(f)
        lats_mg = pickle.load(f)


griddata_plot(dem, lons_mg, lats_mg, "01 A digital elevation model (DEM) of Campi Flegrei.  ")
water_mask = ma.getmask(dem)                                                                    # DEM has no values for water, and we have no radar return from water, so good to keep a mask available



#%% Second, make a deformation signal
signals_m = {}                                                                              # these should be in metres

ll_extent = [(lons_mg[-1,0], lats_mg[-1,0]), (lons_mg[1,-1], lats_mg[1,-1])]                # [lon lat tuple of lower left corner, lon lat tuple of upper right corner]

los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'mogi', dem, **mogi_kwargs)
signals_m["deformation"] = ma.array(los_grid, mask = water_mask)
griddata_plot(signals_m["deformation"], lons_mg, lats_mg, "06 Deformaiton signal", dem_mode = False)                         # plot



#%% make a topograhically correlated atmospheric phase screen (APS), using the DEM
signals_m['topo_correlated_APS'] = atmosphere_topo(dem, strength_mean = 56.0, strength_var = 12.0, difference = True)                    # dem must be in metres
griddata_plot(signals_m["topo_correlated_APS"], lons_mg, lats_mg, "Topographically correlated APS", dem_mode = False)                         # plot
  

#%% Note that we can also synthesise areas of incoherece and use these to update the mask
coherence_mask = coherence_mask(lons_mg, lats_mg, threshold=0.7)                                                                # if threshold is 0, all of the pixels are incoherent , and if 1, none are.  

matrix_show(coherence_mask)

mask_combined = np.logical_or(water_mask, coherence_mask)                                                                       # combine the masks for water and incoherence
# ifg_with_incoherence = ma.array(ma.getdata(signals_m["combined"]), mask = mask_combined)
# griddata_plot(ifg_with_incoherence, lons_mg, lats_mg, "10 Synthetic interferogram with incoherent regions", dem_mode = False)                         # plot
    

#%% Lets make a time series

matrix_show(signals_m["deformation"])

S = np.vstack((ma.compressed(signals_m["deformation"]), ma.compressed(signals_m["topo_correlated_APS"])))         # signals will be stored as row vectors 

ph_turb_m  = atmosphere_turb(n_interferograms, lons_mg, lats_mg, water_mask=water_mask, mean_m = 0.02)
N = np.zeros((n_interferograms, S.shape[1]))
for row_n, ph_turb in enumerate(ph_turb_m):                                         # conver the noise (turbulent atmosphere) into a matrix of row vectors.  
    N[row_n,] = ma.compressed(ph_turb)

A = np.random.randn(n_interferograms,2)                                             # these column vectors control the strength of each source through time

quick_linegraph(A.T)

X = A@S + N                                                                         # do the mixing: X = AS + N

X = A[:,0:1] @ S[0:1,:]                                                                         # do the mixing: X = AS + N


for ifg_n, ifg in enumerate(X):
    matrix_show(col_to_ma(water_mask, ifg), title = f"ifg {ifg_n}")

col_to_ma(water_mask, X[0,:])

plot_ifgs(X, water_mask, title = 'Synthetic Interferograms')    
    
A[:,0] *= 3                                                                         # increase the strength of the first column, which controls the deformation
X = A@S + N                                                                         # do the mixing: X = AS + N
plot_ifgs(X, water_mask, title = 'Synthetic Interferograms: increased deformation strength')


#%%





