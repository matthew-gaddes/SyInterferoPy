#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:55:59 2020

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/')
sys.path.append('./lib/')



from dem_tools_lib import SRTM_dem_make_batch
from random_generation_functions import create_random_synthetic_ifgs

#%% 0: Things to set

SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',                                      # 1 arc second (SRTM1) is not yet supported.  
                     'water_mask_resolution'    : 'f',                                          # 'c', 'i', 'h' or 'f' (lowest to highest resolution, highest can be very slow)
                     'SRTM3_tiles_folder'       : './SRTM3/',                                   # folder for DEM tiles.  
                     'download'                 : True,                                         # If tile is not available locally, try to download it
                     'void_fill'                : True}                                         # some tiles contain voids which can be filled (slow)

synthetic_ifgs_settings = {'defo_sources'           :  ['no_def', 'dyke', 'sill', 'mogi'],      # deformation patterns that will be included in the dataset.  
                           'n_ifgs'                 : 7,                                        # the number of synthetic interferograms to generate
                           'n_pix'                  : 224,                                      # number of 3 arc second pixels (~90m) in x and y direction
                           'outputs'                : ['uuu', 'uud', 'www', 'wwd', 'rid'],     # channel outputs.  uuu = unwrapped across all 3, uu
                           'intermediate_figure'    : False,                                     # if True, a figure showing the steps taken during creation of each ifg is displayed.  
                           'coh_scale'              : 5000,                                     # The length scale of the incoherent areas, in meters.  A smaller value creates smaller patches, and a larger one creates larger pathces.  
                           'coh_threshold'          : 0.7,                                      # if 1, there are no areas of incoherence, if 0 all of ifg is incoherent.  
                           'min_deformation'        : 0.05,                                     # deformation pattern must have a signals of at least this many metres.  
                           'max_deformation'        : 0.25,                                     # deformation pattern must have a signal no bigger than this many metres.  
                           'snr_threshold'          : 2.0,                                      # signal to noise ratio (deformation vs turbulent and topo APS) to ensure that deformation is visible.  A lower value creates more subtle deformation signals.
                           'turb_aps_mean'          : 0.02,                                     # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
                           'turb_aps_length'        : 5000}                                     # turbulent APS will be correlated on this length scale, in metres.  
                           

volcano_dems = [{'name': 'Vulsini',                 'centre': (11.93, 42.6),        'side_length' : (40e3, 40e3)},                               # centre is lon then lat, side length is x then y, in km.  
                {'name': 'Campi Flegrei',           'centre': (14.139, 40.827),     'side_length' : (40e3, 40e3)},
                {'name': 'Witori',                  'centre': (150.516, -5.576),    'side_length' : (40e3, 40e3)},
                {'name': 'Lolobau',                 'centre': (151.158, -4.92),     'side_length' : (40e3, 40e3)},
                {'name': 'Krakatau',                'centre': (105.423, -6.102),    'side_length' : (40e3, 40e3)},
                {'name': 'Batur',                   'centre': (115.375, -8.242),    'side_length' : (40e3, 40e3)},
                {'name': 'Taal',                    'centre': (120.993, 14.002),    'side_length' : (40e3, 40e3)},
                {'name': 'Aira',                    'centre': (130.657, 31.593),    'side_length' : (40e3, 40e3)},
                {'name': 'Asosan',                  'centre': (131.104, 32.884),    'side_length' : (40e3, 40e3)},
                {'name': 'Naruko',                  'centre': (140.734, 38.729),    'side_length' : (40e3, 40e3)},
                {'name': 'Towada',                  'centre': (140.88, 40.51),      'side_length' : (40e3, 40e3)}]
                



#%% 1: Create a list of locations (in this case subaerial volcanoes) to make interferograms for, and make them.  

np.random.seed(0)                                                                                           # 0 used in the example

try:
    print('Trying to open a .pkl of the DEMs... ', end = '')
    with open('example_03_dems.pkl', 'rb') as f:
        volcano_dems2 = pickle.load(f)
    f.close()
    print('Done.  ')

except:
    print('Failed.  Generating them from scratch, which can be slow.  ')
    volcano_dems2 = SRTM_dem_make_batch(volcano_dems, **SRTM_dem_settings)                                  # make the DEMS
    with open(f'example_03_dems.pkl', 'wb') as f:
        pickle.dump(volcano_dems2, f)
    print('Saved the dems as a .pkl for future use.  ')


#%% 2: Create the synthetic interferograms
        
X_all, Y_class, Y_loc = create_random_synthetic_ifgs(volcano_dems2, **synthetic_ifgs_settings)

#%% Plot one interferogram in the different channel formats.  


ifg_n = 3
units = np.array([['rad', 'rad', 'rad', 'rad', 'intensity'],
                  ['rad', 'rad', 'rad', 'rad', 'intensity'],
                  ['rad', 'm',   'rad', 'm',    'm']])


fig1, axes = plt.subplots(3,5, figsize = (14,7))
fig1.subplots_adjust(hspace = 0.4, left = 0.02, right = 0.98, bottom = 0.05, top = 0.9)
fig1.canvas.set_window_title('Channel format of data')
fig1.suptitle('u:unwrapped      d:DEM    w:wrapped      r:real       i:imaginary')

for format_n, key in enumerate(X_all):
    axes[0, format_n].set_title(key)
    for channel_n in range(3):
        image = axes[channel_n, format_n].imshow(X_all[key][ifg_n,:,:,channel_n])
        cbar = plt.colorbar(image, ax = axes[channel_n, format_n], orientation = 'horizontal', pad = 0.2 )
        cbar.ax.set_xlabel(units[channel_n, format_n])
        if format_n == 0:
            axes[channel_n, format_n].set_ylabel(f'Channel {channel_n}')

