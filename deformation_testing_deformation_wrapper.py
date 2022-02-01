#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:07:54 2021

@author: matthew
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/matthew/university_work/python_stuff/python_scripts")                    # personal imports.  
from small_plot_functions import matrix_show

sys.path.append('./lib/')
from syinterferopy_functions import deformation_wrapper


#%%

deformation_ll = (14.14, 40.84,)                                                     # deformation lon lat

lons = np.linspace(14.02143, 14.2577, 285)
lats = np.linspace(40.74995, 40.929215, 216)

lons_mg = np.repeat(lons[np.newaxis,:], lats.shape, axis = 0)
lats_mg = np.flipud(np.repeat(lats[:, np.newaxis], lons.shape, axis = 1))

ll_extent = [(lons_mg[-1,0], lats_mg[-1,0]), (lons_mg[1,-1], lats_mg[1,-1])]                # [lon lat tuple of lower left corner, lon lat tuple of upper right corner]

dem = np.random.randn(lons_mg.shape[0], lons_mg.shape[1])

#%% With a dem, we can now generate geocoded deformation patterns.  

# # 1: Mogi
# los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'mogi', dem, **mogi_kwargs)
# griddata_plot(los_grid, lons_mg, lats_mg, '02: Point (Mogi) source', dem_mode = False)
# matrix_show(ma.getdata(los_grid), title = 'Mogi')


# # 2: Dyke
# dyke_kwargs = {'strike' : 0,                                    # in degrees
#                'top_depth' : 1000,                              # in metres
#                'bottom_depth' : 3000,
#                'length' : 5000,
#                'dip' : 80,                                      # in degrees
#                'opening' : 0.5}
# los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'dyke', dem, **dyke_kwargs)
# #griddata_plot(los_grid, lons_mg, lats_mg, '03: Opening Dyke', dem_mode = False)
# matrix_show(ma.getdata(los_grid), title = 'dyke')

# #3: Sill
# sill_kwargs = {'strike' : 0,                            # degrees
#                'depth' : 3000,                          # metres
#                'width' : 5000,
#                'length' : 5000,
#                'dip' : 1,                               # degrees
#                'opening' : 0.5}
# los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'sill', dem, **sill_kwargs)
# #griddata_plot(los_grid, lons_mg, lats_mg, '04: Inflating Sill', dem_mode = False)
# matrix_show(ma.getdata(los_grid), title = 'sill')


# #4: SS EQ
# quake_ss_kwargs  = {'strike' : 0,                       # degrees
#                     'dip' : 80,
#                     'length' : 5000,                        # metres
#                     'rake' : 0,                             # ie left lateral, 180 for right lateral.  
#                     'slip' : 1,
#                     'top_depth' : 4000,
#                     'bottom_depth' : 8000}
# los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'quake', dem, **quake_ss_kwargs)
# #griddata_plot(los_grid, lons_mg, lats_mg, '05: SS fault earthquake - is magnitude correct?', dem_mode = False)
# matrix_show(ma.getdata(los_grid), title = 'ss')

# #5: Normal EQ
# quake_normal_kwargs  = {'strike' : 0,
#                         'dip' : 70,
#                         'length' : 5000,
#                         'rake' : -90,
#                         'slip' : 1,
#                         'top_depth' : 4000,
#                         'bottom_depth' : 8000}
# los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'quake', dem, **quake_normal_kwargs)
# #griddata_plot(los_grid, lons_mg, lats_mg, '06: Normal fault earthquake - is magnitude correct?', dem_mode = False)
# matrix_show(ma.getdata(los_grid), title = 'normal')

#6: Thrust EQ
quake_thrust_kwargs = {'strike' : 0,
                       'dip' : 30,
                       'length' : 5000,
                       'rake' : 90,
                       'slip' : 1,
                       'top_depth' : 4000,
                       'bottom_depth' : 8000}
los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'quake', dem, **quake_thrust_kwargs)
#griddata_plot(los_grid, lons_mg, lats_mg, '07: Thurst fault earthquake - is magnitude correct?', dem_mode = False)        
        
matrix_show(ma.getdata(los_grid), title = 'thrust')
# matrix_show(ma.getdata(x_grid), title = 'thrust - x')
# matrix_show(ma.getdata(y_grid), title = 'thrust - y')
# matrix_show(ma.getdata(z_grid), title = 'thrust - z')

        

        
