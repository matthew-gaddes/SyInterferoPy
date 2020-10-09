#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:55:59 2020

SyInterferoPy can also make random interferograms, which can be espcially useful for training deep learning models (E.g. Convolutional neural networks)

@author: matthew

CURRENT STATUS:
    IT APPEARS THAT LATS ARE POSSIBLY NOT BEING HANDLED CORRECTLY EITHER SIDE OF THE EQUATOR (NORTHERNMOST SHOULD 
                                                                                              ALWAYS BE AT THE TOP)
    try to remake the DEMS to see if this fixes it.  
"""
import numpy as np
import pickle
import sys
sys.path.append('/home/matthew/university_work/11_DEM_tools/SRTM-DEM-tools/')
sys.path.append('./lib/')



from dem_tools_lib import SRTM_dem_make_batch
from random_generation_functions import create_random_synthetic_ifgs

#%% 0: Things to set

SRTM_dem_settings = {'SRTM1_or3'                : 'SRTM3',
                     'water_mask_resolution'    : 'f',
                     'SRTM3_tiles_folder'       : './SRTM3/',
                     'download'                 : True,
                     'void_fill'                : True}

synthetic_ifgs_settings = {'defo_sources'           :  ['mogi'], #['no_def', 'dyke', 'sill', 'mogi'],
                           'n_ifgs'                 : 10,
                           'n_pix'                  : 224,
                           'outputs'                : ['uuu'],
                           'intermediate_figure'    : True,
                           'coh_scale'              : 5000,
                           'coh_threshold'          : 0.7,
                           'min_deformation'        : 0.05,
                           'max_deformation'        : 0.25,
                           'snr_threshold'          : 2.0,
                           'turb_aps_mean'          : 0.02,                         # turbulent APS will have, on average, a maximum strenghto this in metres (e.g 0.02 = 2cm)
                           'turb_aps_length'        : 5000}                         # turbulent APS will be correlated on this length scale, in metres.  
                           

volcano_dems = [{'name': 'Vulsini',                 'centre': (11.93, 42.6),        'side_length' : (40e3, 40e3)},                               # centre is lon then lat, side length is x then y, in km.  
                {'name': 'Campi Flegrei',           'centre': (14.139, 40.827),     'side_length' : (40e3, 40e3)},
                {'name': 'Tofua',                   'centre': (-175.07, -19.75),    'side_length' : (40e3, 40e3)},
                {'name': 'Witori',                  'centre': (150.516, -5.576),    'side_length' : (40e3, 40e3)},
                {'name': 'Lolobau',                 'centre': (151.158, -4.92),     'side_length' : (40e3, 40e3)},
                {'name': 'Kuwae',                   'centre': (168.536, -16.829),   'side_length' : (40e3, 40e3)},
                {'name': 'Krakatau',                'centre': (105.423, -6.102),    'side_length' : (40e3, 40e3)},
                {'name': 'Batur',                   'centre': (115.375, -8.242),    'side_length' : (40e3, 40e3)},
                {'name': 'Banda Api',               'centre': (129.881, -4.523),    'side_length' : (40e3, 40e3)},
                {'name': 'Taal',                    'centre': (120.993, 14.002),    'side_length' : (40e3, 40e3)},
                {'name': 'Kikai',                   'centre': (130.308, 30.789),    'side_length' : (40e3, 40e3)},
                {'name': 'Aira',                    'centre': (130.657, 31.593),    'side_length' : (40e3, 40e3)},
                {'name': 'Asosan',                  'centre': (131.104, 32.884),    'side_length' : (40e3, 40e3)},
                {'name': 'Naruko',                  'centre': (140.734, 38.729),    'side_length' : (40e3, 40e3)},
                {'name': 'Towada',                  'centre': (140.88, 40.51),      'side_length' : (40e3, 40e3)},
                {'name': 'Nishinoshima',            'centre': (140.874, 27.247),    'side_length' : (40e3, 40e3)},
                {'name': 'Ioto',                    'centre': (141.289, 24.751),    'side_length' : (40e3, 40e3)}]
                

#%%






#%% 1: Create a list of locations (in this case subaerial volcanoes) to make interferograms for, and make them.  


np.random.seed(0)

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

#%%  testing that the lons and lats look ok
# import matplotlib.pyplot as plt

# for volcano_dem2 in volcano_dems2:
#     f, axes = plt.subplots(1,2)
#     f.suptitle(volcano_dem2['name'])
#     im = axes[0].imshow(volcano_dem2['lons_mg'])
#     f.colorbar(im, ax = axes[0])
#     im = axes[1].imshow(volcano_dem2['lats_mg'])
#     f.colorbar(im, ax = axes[1])
    


#%%

# import numpy.ma as ma
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.pylab import *

# ph_defo = volcano_dems2[0]['dem'][:432, :569]

# mask = ma.getmask(volcano_dems2[1]['dem'][:432, :569]).astype(int)

# f, ax = plt.subplots(1)
# #ax.imshow(ma.array(ph_defo, mask = mask))
# ax.imshow(ph_defo)



# # x = np.arange(0, ph_defo.shape[1])
# # y = np.arange(0, ph_defo.shape[0])

# # cs = ax.contourf(x, y, mask, 1 , hatches=['', '/'],  alpha=1)

# plt.contourf(mask, 1, hatches=['', '//'], alpha=0)


# #%%
# # x_mg, y_mg = np.meshgrid(np.arange(0, test.shape[1]), np.arange(0, test.shape[0]))

# # x, y = x_mg.flatten(), y_mg.flatten()
# #z = ma.getmask(test).flatten()
# z = ma.getmask(test).astype(int)                    # mask as 1 and 0s

# [m,n] = np.where(z > 0.5)                           # split to get points we want

# z1=np.zeros(z.shape)
# z1[m, n] = 99

# # use contourf() with proper hatch pattern and alpha value
# cs = ax.contourf(x, y, ph_defo ,3 , hatches=['', '/'],  alpha=1)

# #cs = ax.contourf(x, y, z, hatches=['-', '/'], alpha = 0.5, cmap = 'gray')                            # , '/', '\\', '//']
#                   #cmap='gray', extend='both', alpha=0.5)



# # for i, j in np.argwhere(ma.getmask(test) == True)[:10,:]:
# #     ax.add_patch(mpl.patches.Rectangle((i-.5, j-.5), 1, 1, hatch='/', fill=False, snap=False))

# # for i,j in np.floor(50*rand(10,2)).astype('int'):
# #     ax.add_patch(mpl.patches.Rectangle((i-.5, j-.5), 1, 1, hatch='///////', fill=False, snap=False))



# import sys; sys.exit()

#%% 2: Create the synthetic interferograms
        
X_all, Y_class, Y_loc = create_random_synthetic_ifgs(volcano_dems2, **synthetic_ifgs_settings)
