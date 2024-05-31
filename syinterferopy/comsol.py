#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:11:50 2024

@author: matthew
"""

import pdb 

#%%

def sypy_dem_to_comsol_dem(dem, lons_mg, lats_mg, outfile = None):
    """ Given a SyInterferoPy DEM and the lons and lats of each pixel in it,
    conver to COMSOL form (text file, with x y z in metres as rows of a text 
   file).  
    
    Inputs:
        dem | rank 2 array | heigh of each pixel in metres
        lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
        lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
    
    Returns:
        ijk | rank 2 array | 3x lots.  The distance of each pixel from the lower 
                            left corner of the image in metres.  
                            
    History:
        2024_05_31 | MEG | Written.  
    
    """
    import numpy as np
    from syinterferopy.aux import lon_lat_to_ijk
    
    # get the pixel positions relative to bottom left corner in metres
    ijk, pixel_spacing = lon_lat_to_ijk(lons_mg, lats_mg)
    
    # convert flat surface (k = 0) to DEM heights.  
    ijk[2,:] = np.ravel(dem)
    
    # # debug plotting function
    # f, ax = plt.subplots()
    # ax.scatter(ijk[0,:], ijk[1,:], c = ijk[2,:] )
    
    # possibly also write to a file
    if outfile is not None:
        with open(outfile, 'w') as file:
         # Iterate over the rows of the array
         for row in ijk.T:

             # Convert each row to a string and join the elements with a space
             line = ' '.join(map(str, row))
             # Write the line to the file
             file.write(line + '\n')

    return ijk
    

#%%