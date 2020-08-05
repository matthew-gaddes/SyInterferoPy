#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 14:53:45 2020

@author: matthew
"""



#%% simple case, using model_generator.m


from oct2py import Oct2Py
oc = Oct2Py(temp_dir="/home/matthew/university_work/01_blind_signal_separation_python/14_Synthetic_Interferograms_GitHub/m_files/octave_temp_dir/")


oc.addpath("/home/matthew/university_work/01_blind_signal_separation_python/14_Synthetic_Interferograms_GitHub/m_files/")

# 1 from docs
# import numpy as np
# x = np.array([[1, 2], [3, 4]], dtype=float)
# out, oclass = oc.roundtrip(x)

# 2 ?
test = oc.model_generator(10, 4, './outputs/dyke.mat', 324, 0.02, 0.25)
#los_all, source_parameters = oc.model_generator(10, 4, './outputs/dyke.mat', 324, 0.02, 0.25)

# 3 ?
#los_all, source_parameters = oc.model_generator(10, 4, 'outputs/dyke.mat', 324, 0.02, 0.25)

#%%

#[los_all, source_parameters] = model_generator(n_models, Source_Type, filename, n_pixs, min_disp_required, max_disp_required);

def eq_dyke_sill_deformation(source, n_pixs = 324, m_in_pix = 90, asc_or_desc = 'asc', incidence = 23,
                             **kwargs):    
    """
    A function to create deformation patterns for either an earthquake, dyke or sill.  
    This functions calls disloc3d4, which then calls dc3d4 (earthquake), dc3d5 (dyke), or dc3d6 (sill).  
    To aid in readability, different sources take different parameters (e.g. slip for a quake, opening for a dyke)
    are passed separately as kwargs, even if they ultimately go into the same field in the model parameters.  
    
    A quick recap on definitions:
        strike - measured clockwise from 0 at north, 180 at south.  fault dips to the right of this.  hanging adn fo
        dip - measured from horizontal, 0 for horizontal, 90 for vertical.  
        rake - direction the hanging wall moves during rupture, measured relative to strike, anticlockwise is positive, so:
            0 for left lateral ss 
            180 (or -180) for right lateral ss
            -90 for normal
            90 for thrust

    Inputs:
        source | string | quake or dyke or sill
        n_pixs | int | rank 2 arrays returned will be square, and with this many pixels on each side.  
        m_in_pix | int | sets the size of a pixel.  defaults is 90m for SRTM3 resolution.  
        asc_or_dec | string | 'asc' or 'desc' or 'random'.  If set to 'random', 50% chance of each.  
        incidence | float | satellite incidence angle.  
        
        examples of kwargs:
            
        quake_normal = {'strike' : 0,
                        'dip' : 70,
                        'length' : 5000,
                        'rake' : -90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_thrust = {'strike' : 0,
                        'dip' : 30,
                        'length' : 5000,
                        'rake' : 90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_ss = {'strike' : 0,
                    'dip' : 80,
                    'length' : 5000,
                    'rake' : 0,
                    'slip' : 1,
                    'top_depth' : 4000,
                    'bottom_depth' : 8000}
        
        dyke = {'strike' : 0,
                'top_depth' : 1000,
                'bottom_depth' : 3000,
                'length' : 5000,
                'dip' : 80,
                'opening' : 0.5}
        
        sill = {'strike' : 0,
                'depth' : 3000,
                'width' : 5000,
                'length' : 5000,
                'dip' : 1,
                'opening' : 0.5}
        

    Returns:
        
    History:
        2020/08/05 | MEG | Written
    """    
    import numpy as np
    from oct2py import Oct2Py  
    
    oc = Oct2Py(temp_dir="/home/matthew/university_work/01_blind_signal_separation_python/14_Synthetic_Interferograms_GitHub/m_files/octave_temp_dir/")
    oc.addpath("/home/matthew/university_work/01_blind_signal_separation_python/14_Synthetic_Interferograms_GitHub/m_files/")
    
       
    # 1:  Setting for heading and elastic parameters.  
    if asc_or_desc == 'asc':
        heading = 192.04
    elif asc_or_desc == 'desc':
        heading = 012.04
    elif asc_or_desc == 'random':
        if (-0.5+np.random.rand()) < 0.5:
            heading = 192.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are descending
        else:
            heading = 012.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are ascending
    else:
        raise Exception(f"'asc_or_desc' must be either 'asc' or 'desc' or 'random', but is currently {asc_or_desc}.  Exiting.   ")
    
    lame = {'lambda' : 2.3e10,                                                         # elastic modulus (Lame parameter, units are pascals)
            'mu'     : 2.3e10}                                                         # shear modulus (Lame parameter, units are pascals)
    v = lame['lambda'] / (2*(lame['lambda'] + lame['mu']));                         #  calculate poisson's ration
            
    # 2:  Calculate LOS_vector from Heading and Incidence
    deg2rad = np.pi/180
    sat_inc = 90 - incidence
    sat_az  = 360 - heading
    #sat_inc=Incidence                                                      # hmmm - why the different definition?
    #sat_az=Heading;        
    los_x=-np.cos(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_y=-np.sin(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_z=np.sin(sat_inc*deg2rad);
    los_vector = np.array([[los_x],
                           [los_y],
                           [los_z]])                                          # Unit vector in satellite line of site    
        
        
    # 2: set up regular grid
    min_max = int((m_in_pix * (n_pixs-1)) / 2)                                    # clcaulte number of m to go in each direction to get the right number of pixels
    x = np.arange(-min_max, min_max+1, m_in_pix)
    y = np.arange(-min_max, min_max+1, m_in_pix)
    xx, yy = np.meshgrid(x, y)
    xx = np.ravel(xx)
    yy = np.ravel(yy)
    coords = np.vstack((xx[np.newaxis,:], yy[np.newaxis,:]))                        # x is top row, y is bottom row
    
    
    # 3: set 10x1 of model parameters needed by disloc3d4, and call
    model = np.zeros((10,1))
    model[0,0] = 1                                                  # some parameters are shared for all three types and can be set here
    model[1,0] = 1
    model[2,0] = kwargs['strike']
    model[3,0] = kwargs['dip']
    model[6,0] = kwargs['length']
    if source == 'quake':                                           # otherwise they need to be set for each type
        model[4,0] = kwargs['rake']
        model[5,0] = kwargs['slip']
        model[7,0] = kwargs['top_depth']
        model[8,0] = kwargs['bottom_depth']
        model[9,0] = 1
    elif source == 'dyke':
        model[4,0] = 0
        model[5,0] = kwargs['opening']
        model[7,0] = kwargs['top_depth']
        model[8,0] = kwargs['bottom_depth']
        model[9,0] = 2    
    elif source == 'sill':
        model[4,0] = 0
        model[5,0] = kwargs['opening']
        model[7,0] = kwargs['depth']
        model[8,0] = kwargs['width']
        model[9,0] = 3        
    else:
        raise Exception(f"'Source' must be either 'quake', 'dyke', or 'sill', but is set to {source}.  Exiting.")
    
    U = oc.disloc3d4(model, coords, lame['lambda'], lame['mu'])                           # U is 3xnpoints and is the displacement in xyz directions.  
    
    # 5: change from xyz deformation to in satellite LOS
    xgrid = np.reshape(U[0,], (len(y), len(x)))
    ygrid = np.reshape(U[1,], (len(y), len(x)))
    zgrid = np.reshape(U[2,], (len(y), len(x)))
    los_grid = xgrid*los_vector[0,0] + ygrid*los_vector[1,0] + zgrid*los_vector[2,0]
    
    return xgrid, ygrid, zgrid, los_grid




dyke = {'strike' : 0,
        'top_depth' : 1000,
        'bottom_depth' : 3000,
        'length' : 5000,
        'dip' : 80,
        'opening' : 0.5}

xgrid, ygrid, zgrid, los_grid = eq_dyke_sill_deformation('dyke', **dyke) 
                                                          


import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.imshow(los_grid)





#%% more complex case, using model_generator in python

#[los_all, source_parameters] = model_generator(n_models, Source_Type, filename, n_pixs, min_disp_required, max_disp_required);
"""
A function to create deformation patterns for either an earthquake, dyke or sill
This could be called earthquake_dyke_sill_deformation?   Would then need to set parameters.  
Random generation can be set in a funtion that calls this one.  ?
This functions calls disloc3d4, which then calls dc3d4 (earthquake), dc3d5 (dyke), or dc3d6 (sill)

A quick recap on definitions:
    strike - measured clockwise from 0 at north, 180 at south.  fault dips to the right of this.  hanging adn fo
    dip - measured from horizontal, 0 for horizontal, 90 for vertical.  
    rake - direction the hanging wall moves during rupture, measured relative to strike, anticlockwise is positive, so:
        0 for left lateral ss 
        180 (or -180) for right lateral ss
        -90 for normal
        90 for thrust
    

Earthquake parameters:
    strike
    top_depth
    bottom_depth
    length
    slip
    dip                         [45, 70] for ss?, [70, 90] for normal?, [25, 50] for thrust   NB SS and normal appear reversed?
    rake                        [0, 10] for ss, [-110 -70] for normal, [70, 110] for thrust
    
Dyke parameters:
    strike
    top_depth
    bottom_depth
    length
    dip
    opening
    
Sill parameters:
    strike
    depth
    width
    length
    dip
    opening
    
    
    


% type = 1;                % Rectangular dislocatoin: for ss eq
% type = 2                % Rectangular dislocation: normal fault eq
% type = 3                % Rectangular dislocation: thrust fault eq
% type = 4                % Rectangular dislocation: opening dyke
% type = 5                % Rectangular dislocation: opening Sill
% type = 6                % Point pressure source (Mogi)
% type = 7                % Penny shaped horizontal crack (Fialko)

"""

quake_normal = {'strike' : 0,
                'dip' : 70,
                'length' : 5000,
                'rake' : -90,
                'slip' : 1,
                'top_depth' : 4000,
                'bottom_depth' : 8000}

quake_thrust = {'strike' : 0,
                'dip' : 30,
                'length' : 5000,
                'rake' : 90,
                'slip' : 1,
                'top_depth' : 4000,
                'bottom_depth' : 8000}

quake_ss = {'strike' : 0,
            'dip' : 80,
            'length' : 5000,
            'rake' : 0,
            'slip' : 1,
            'top_depth' : 4000,
            'bottom_depth' : 8000}

    
# function args
source = 'quake'
n_pixs = 324
kwargs = quake_ss



# end function args


import numpy as np
from oct2py import Oct2Py
from oct2py import Struct 



#% Something to check if the correct args have been passed?



#% Settings (Heading and Incidence Angle for Satellite, Elastic Lame parameters
if (-0.5+np.random.rand()) < 0.5:
    heading = 192.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are descending
else:
    heading = 012.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are ascending
incidence = 23                                                                  # Incidence angle of satellite in degrees

lame = {'lambda' : 2.3e10,                                                         # elastic modulus (Lame parameter, units are pascals)
        'mu'     : 2.3e10}                                                         # shear modulus (Lame parameter, units are pascals)
v = lame['lambda'] / (2*(lame['lambda'] + lame['mu']));                         #  calculate poisson's ration

m_in_pix = 90;                                                                 # for SRTM3 resolution
    
#% Calculate LOS_vector from Heading and Incidence
deg2rad = np.pi/180
sat_inc = 90 - incidence
sat_az  = 360 - heading
#sat_inc=Incidence                                                      # hmmm - why the different definition?
#sat_az=Heading;        
los_x=-np.cos(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
los_y=-np.sin(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
los_z=np.sin(sat_inc*deg2rad);
los_vector = np.array([[los_x],
                       [los_y],
                       [los_z]])                                          # Unit vector in satellite line of site    
    
    
#% set up regular grid
min_max = int((m_in_pix * (n_pixs-1)) / 2)                                    # clcaulte number of m to go in each direction to get the right number of pixels
x = np.arange(-min_max, min_max+1, m_in_pix)
y = np.arange(-min_max, min_max+1, m_in_pix)
xx, yy = np.meshgrid(x, y)
xx = np.ravel(xx)
yy = np.ravel(yy)
coords = np.vstack((xx[np.newaxis,:], yy[np.newaxis,:]))                        # x is top row, y is bottom row

#%los_all = np.zeros()

oc = Oct2Py(temp_dir="/home/matthew/university_work/01_blind_signal_separation_python/14_Synthetic_Interferograms_GitHub/m_files/octave_temp_dir/")
oc.addpath("/home/matthew/university_work/01_blind_signal_separation_python/14_Synthetic_Interferograms_GitHub/m_files/")
#test = oc.model_generator(10, 4, './outputs/dyke.mat', 324, 0.02, 0.25)

# great the 10x1 of model parameters needed by disloc3d4
model = np.zeros((10,1))
model[0,0] = 1                                                  # some parameters are shared for all three types and can be set here
model[1,0] = 1
model[2,0] = kwargs['strike']
model[3,0] = kwargs['dip']
model[6,0] = kwargs['length']
if source == 'quake':                                           # otherwise they need to be set for each type
    model[4,0] = kwargs['rake']
    model[5,0] = kwargs['slip']
    model[7,0] = kwargs['top_depth']
    model[8,0] = kwargs['bottom_depth']
    model[9,0] = 1
elif source == 'dyke':
    model[4,0] = 0
    model[5,0] = kwargs['opening']
    model[7,0] = kwargs['top_depth']
    model[8,0] = kwargs['bottom_depth']
    model[9,0] = 2    
elif source == 'sill':
    model[4,0] = 0
    model[5,0] = kwargs['opening']
    model[7,0] = kwargs['depth']
    model[8,0] = kwargs['width']
    model[9,0] = 3        
else:
    raise Exception(f"'Source' must be either 'quake', 'dyke', or 'sill', but is set to {source}.  Exiting.")



U = oc.disloc3d4(model, coords, lame['lambda'], lame['mu'])                           # U is 3xnpoints and is the displacement in xyz directions.  

xgrid = np.reshape(U[0,], (len(y), len(x)))
ygrid = np.reshape(U[1,], (len(y), len(x)))
zgrid = np.reshape(U[2,], (len(y), len(x)))
los_grid = xgrid*los_vector[0,0] + ygrid*los_vector[1,0] + zgrid*los_vector[2,0]


import matplotlib.pyplot as plt
f, ax = plt.subplots()
ax.imshow(los_grid)

import sys; sys.exit()

#%%
"""

>>> from oct2py import Struct
>>> test = Struct()
>>> test['foo'] = 1
>>> test.bizz['buzz'] = 'bar'
>>> test
{'foo': 1, 'bizz': {'buzz': 'bar'}}
>>> import pickle
>>> p = pickle.dumps(test)

model = [1;
         1;
         Quake.Strike;
         Quake.Dip;
         Quake.Rake;
         Quake.Slip;
         Quake.Length*1000;
         Quake.Top_depth*1000;
         Quake.Bottom_depth*1000;
         1];       


model = [1;
         1;
         Dyke.Strike;
         Dyke.Dip;
         0;
         Dyke.Opening;
         Dyke.Length*1000;
         Dyke.Top_depth*1000;
         Dyke.Bottom_depth*1000;
         2];

model = [1;
         1;
         Sill.Strike;
         Sill.Dip;
         0;
         Sill.Opening;
         Sill.Length*1000;
         Sill.Depth*1000;
         Sill.Width*1000;
         3];


"""

# most of thist stuff to go, including number of models.  


    # #%% 1 strike slip EQs
    # los_all = zeros(n_models, size(x, 2), size(x,2));                           % initate for outputs of loop
    # i = 1;                                                                           %number of succesfully generated models
    # while i < (n_models + 1)  
    # %for i = 1:n_models                                                          % loop through making models
    #     fprintf('Generating model %s of %d models.\n', num2str(i), n_models);

    #     if Source_Type == 1 || Source_Type == 2 || Source_Type == 3 || Source_Type == 4 || Source_Type == 5      % If using disloc
    #         if Source_Type == 1 || Source_Type == 2 || Source_Type == 3                                % SS/N/T eq

    #             % first determine the size of the eq
    #             Mw = 5.4 + 0.6*rand();                                                  % generate moment magnitude of event (4-6)
    #             Mo = 10^((3/2)*(Mw + 6.06));                                        % seismic moment (Nm or J)
    #             eq_const = (20000*(Mo/mu))^(1/3);                                   % crude scaling - square rupture, and slip is 1/20000 of length of patch
    #             eq_const_km = eq_const * (1e-3);                                    % conver to km

    #             % then use size to set correct quake parameters
    #             %Quake.Strike = 0;
    #             Quake.Strike = randi([0,359]);
    #             Quake.Top_depth = 5*rand();
    #             Quake.Bottom_depth = Quake.Top_depth + eq_const_km;                 % patch of height x
    #             Quake.Length = eq_const_km;                                         % width is x (square path)
    #             Quake.Slip = (eq_const/20000);                                      % slip is 1/20000 of slip patch length in m
    #             if Source_Type == 1                                                 % quake params for ss
    #                 Quake.Dip = randi([45,70]);
    #                 Quake.Rake = randi([0, 10]);            
    #             elseif Source_Type == 2                                             % quake params for normal
    #                 Quake.Dip = randi([70,90]);
    #                 Quake.Rake = randi([-110, -70]);            
    #             else                                                                % quake params for thrust
    #                 Quake.Dip = randi([25,50]);
    #                 Quake.Rake = randi([70, 110]);            
    #             end
    #             model = [1;1;Quake.Strike;Quake.Dip;Quake.Rake;Quake.Slip;Quake.Length*1000;Quake.Top_depth*1000;Quake.Bottom_depth*1000;1];       
    #             source_parameters(i) = Quake;                                       % record what generated each model

    #         elseif Source_Type == 4                                     
    #             %Dyke.Strike = 0;                                          %strike in degrees
    #             Dyke.Strike = randi([0,359]);
    #             Dyke.Dip = randi([75, 90]);                                             %dip in degrees (usually 90 or near 90)
    #             Dyke.Opening = 0.1 + 0.6*rand();                                                %magnitude of opening (perpendincular to plane) in metres
    #             Dyke.Top_depth = 2*rand();                                              %depth (measured vertically) to top of dyke in kilometres
    #             Dyke.Bottom_depth = Dyke.Top_depth + 6*rand();                          %depth (measured vertically) to bottom of dyke in kilometres
    #             Dyke.Length = 10*rand() ;                                                %dyke length in kilometres
    #             model = [1;1;Dyke.Strike;Dyke.Dip;0;Dyke.Opening;Dyke.Length*1000;Dyke.Top_depth*1000;Dyke.Bottom_depth*1000;2];
    #             source_parameters(i) = Dyke;                                            % record what generated each model

    #         elseif Source_Type == 5
    #             %Sill.Strike = 0;                                  %strike (orientation of Length dimension) in degrees
    #             Sill.Strike = randi([0,359]);
    #             Sill.Dip = randi([0,5]);                                        %Dip in degrees (usually zero or near zero)
    #             Sill.Opening = 0.2 + 0.8*rand();                                        %magnitude of opening (perpendincular to plane) in metres
    #             Sill.Depth = 1.5 + 2*rand();                                          %depth (measured vertically) to top of dyke in kilometres
    #             Sill.Width = 2 + 4*rand();                                          %
    #             %Sill.Length = 5*rand();                                                %dyke length in kilometres
    #             %Sill.Length = (0.75 + 0.5*rand) * Sill.Width;                          %dyke length in kilometres
    #             Sill.Length = 2 + 4*rand();                                             %dyke length in kilometres
    #             model = [1;1;Sill.Strike;Sill.Dip;0;Sill.Opening;Sill.Length*1000;Sill.Depth*1000;Sill.Width*1000;3];
    #             source_parameters(i) = Sill;                                        % record what generated each model

    #         end

    #         # this is main part to keep.  

    #         [U,flag]=disloc3d4(model,coords,lambda,mu);                                     % get surface displacements in 3D
    #         xgrid = reshape(U(1,:),numel(y),numel(x));                                      % covnert from row vectors to rank 2
    #         ygrid = reshape(U(2,:),numel(y),numel(x));
    #         zgrid = reshape(U(3,:),numel(y),numel(x));
    #         los_grid = xgrid*LOS_vector(1) + ygrid*LOS_vector(1) + zgrid*LOS_vector(3);








