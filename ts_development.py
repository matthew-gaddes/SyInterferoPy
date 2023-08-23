#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:14:02 2019

@author: mgaddes
"""
#%%


import numpy as np
import numpy.ma as ma
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
# plt.ion()
# try:
#     %matplotlib qt
# except:
#     print(f"Tried ipython magic (%matplotlib qt) to make figures open in windows, but this failed for an unknown reason.  Continuing anyway.  ")


import pickle
import pdb


import syinterferopy
from syinterferopy.syinterferopy import atmosphere_topo, atmosphere_turb, deformation_wrapper, coherence_mask
from syinterferopy.aux import col_to_ma, griddata_plot, plot_ifgs                        

#%% debugging functions

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



#%%



def generate_random_temporal_baselines(d_start = "20141231", d_stop = "20230801"):
    """ Given a date range, generate LiCSAR style short temporal baseline ifgs.  Takes into account that S1b was operational and 
    there were more 6 day ifg then.  
    Inputs:
        d_start | str | YYYYMMDD of when to start time series
        d_stop | str | YYYYMMDD of when to stop time series
    Returns:
        acq_dates | list of datetimes | acq dates.  
        tbaselines | list of ints | timeporal baselines of short temporal baseline ifgs.  First one is 0.  
    History:
        2023_08_17 | MEG | Written
    """
    usual_tbaselines = [6, 12, 12, 12, 12, 12, 12, 12, 12, 12,                  # temp baseline is chosen from this list at random
                         24, 24, 24, 24, 24, 24, 36, 48, 60, 72]                # If number features more, will be chosen more

    from datetime import datetime, timedelta

    # hard coded.  Not exact (i.e. some frames start and stop at different times.)
    s1b_dstart = datetime.strptime("20160901", '%Y%m%d')                       # launch 20160425, ramp up to fully operational over next few months.  
    s1b_dstop = datetime.strptime("20161223", '%Y%m%d')                        # power failure ended mission

    dstart = datetime.strptime(d_start, '%Y%m%d')                              # 
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # 

    
    acq_dates = [dstart]
    tbaselines = [0]
    dcurrent = acq_dates[-1]

    while dcurrent < dstop:
        if (s1b_dstart < dcurrent) and (dcurrent < s1b_dstop ):                  # check if the next date is during the S1b years.  
            tbaseline =  int(np.random.choice(usual_tbaselines[0:]))             # 6 day ifg possible
        else:
            tbaseline = days = int(np.random.choice(usual_tbaselines[1:]))       # 6 day ifg not possible 
    
        dnext = dcurrent + timedelta(days = tbaseline)                           # add the temp baseline to find the new date   
        if dnext < dstop:                                                        # check we haven't gone past the end date 
            acq_dates.append(dnext)                                              # if we haven't, record            
            tbaselines.append(tbaseline)
            dcurrent = acq_dates[-1]                                             # record the current date
        else:
            break                                                                # remember to exit the while if we have got to the last date

    return acq_dates, tbaselines


def tc_uniform_inflation(def_rate = 0.07, d_start = "20141231", d_stop = "20230801"):
    """ Calculate the magnitdue of an inflating signal at each day.  Inflation is linear.  
    Inputs:
        def_rate | float | deformation rate, my/yr
        d_start | str | YYYYMMDD of when to start time series.  Inclusive.  
        d_stop | str | YYYYMMDD of when to stop time series.  Not inclusive.  
    Returns:
        tc_def | list of floats | cumulative deformation on each day.  Note that last day is not included.  
        def_dates | list of dates | dateteime for each day there is deformation for.  Note that last day is not includes
    History:
        2023_08_17 | MEG | Written
    """
    from datetime import datetime, timedelta


    # make cumulative deformaitn for each day
    dstart = datetime.strptime(d_start, '%Y%m%d')                              # conver to dateteime
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # convert to datetime
    n_days = (dstop - dstart).days                                             # find number of days between dates
    max_def = def_rate * (n_days / 365)                                         # calculate the maximum deformation
    tc_def = np.linspace(0, max_def, n_days)                                    # divide it up to calucaulte it at each day
    
    # make datetime for each day.  
    def_dates = [dstart]
    dcurrent = def_dates[-1]

    while dcurrent < dstop:
        dnext = dcurrent + timedelta(days = 1)                                  # advance by one day
        def_dates.append(dnext)
        dcurrent = def_dates[-1]                                                # record the current date
    
    def_dates = def_dates[:-1]                                                  # drop last date
    return tc_def, def_dates


def sample_deformation_on_acq_dates(acq_dates, tc_def, def_dates):
    """ Given deformation on each day (tc_def and def_dates), find the deformation
    on the acquisition days (acq_dates).  
    
    Inputs:
        acq_dates | list of datetimes | acq dates.  
        tc_def | list of floats | cumulative deformation on each day.  Note that last day is not included.  
        def_dates | list of dates | dateteime for each day there is deformation for.  Note that last day is not includes
    Returns:
        tc_def_resampled | r2 numpy array | cumulative deformation on each acquisition day.  
    History:
        2023_08_23 | MEG | Written
    """
    
    n_acq = len(acq_dates)                                                  # 
    tc_def_resampled = np.zeros((n_acq, 1))                                 # initialise as empty (zeros)
    
    for acq_n, acq_date in enumerate(acq_dates):                        
        day_arg = def_dates.index(acq_date)                                 # find which day number the acquiisiont day is
        day_def = tc_def[day_arg]                                           # get the deformaiton for that day
        tc_def_resampled[acq_n, 0] = day_def                                # record
    
    return tc_def_resampled



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


print(f"Generating the turbulent APSs for the time series...", end = '')
ph_turb_m  = atmosphere_turb(n_interferograms, lons_mg, lats_mg, water_mask=water_mask, mean_m = 0.02)
N = np.zeros((n_interferograms, S.shape[1]))
for row_n, ph_turb in enumerate(ph_turb_m):                                         # conver the noise (turbulent atmosphere) into a matrix of row vectors.  
    N[row_n,] = ma.compressed(ph_turb)
print('Done!')


#%%
# # v1:
# A = np.random.randn(n_interferograms,2)                                             # these column vectors control the strength of each source through time
# quick_linegraph(A.T)
# X = A@S + N                                                                         # do the mixing: X = AS + N
# X = A[:,0:1] @ S[0:1,:]                                                             # do the mixing: X = AS + N, but only for deformation.  

# v2: use a function



acq_dates, t_baselines = generate_random_temporal_baselines(d_start = "20141231", d_stop = "20230801")
tc_def, def_dates = tc_uniform_inflation(def_rate = 2., d_start = "20141231", d_stop = "20230801")
tc_def_resampled = sample_deformation_on_acq_dates(acq_dates, tc_def, def_dates)


quick_linegraph([tc_def_resampled])
f, ax = plt.subplots(1,1)
ax.plot(np.cumsum(t_baselines), tc_def_resampled)

#%%



# fig, axes = plt.subplots(4,5)
# for ifg_n, ifg in enumerate(X):
#     np.ravel(axes)[ifg_n].matshow(col_to_ma(ifg, water_mask), vmin = np.min(X), vmax = np.max(X))




# plot_ifgs(X, water_mask, title = 'Synthetic Interferograms')    




