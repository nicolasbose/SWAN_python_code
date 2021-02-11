import numpy as np

def iterable(arr):

    '''Returns an iterable'''
   

    try:
        iter(arr)
        return arr
    except:
        return (arr,)


def expand_and_repeat(mtx, shape=None, repeat=None,
                      exist_dims=None, expand_dims=None):

    '''Expands matrix and repeats matrix contents along new dimensions
    Provide ``shape`` and ``exist_dims`` or ``expand_dims``, or
    ``repeat`` and ``expand_dims``.

    Parameters
    ----------

    mtx : numpy.ndarray
      Input matrix
    shape : tuple, optional
      Target shape of output matrix
    repeat : tuple or int, optional
      Repititions along new dimensions
    exist_dims : tuple or int, optional
      Indices of dimensions in target shape that are present in input matrix
    expand_dims : tuple or int, optional
      Indices of dimensions in target shape that are not present in input matrix

    Returns
    -------
    numpy.ndarray
      Matrix with target shape
    Examples

    --------

    >>> expand_and_repeat([[1,2,3],[4,5,6]], shape=(2,3,4), exist_dims=(0,1))

    >>> expand_and_repeat([[1,2,3],[4,5,6]], shape=(2,3,4), expand_dims=(2,))

    >>> expand_and_repeat([[1,2,3],[4,5,6]], shape=(2,3,4), expand_dims=2)

    >>> expand_and_repeat([[1,2,3],[4,5,6]], repeat=(4,), expand_dims=(2,))

    >>> expand_and_repeat([[1,2,3],[4,5,6]], repeat=4, expand_dims=2)


    '''


    mtx = np.asarray(mtx)
 
    if shape is not None:
        shape = iterable(shape)
   

        if mtx.ndim > len(shape):

            raise ValueError('Nothing to expand. Number of matrix '

                             'dimensions (%d) is larger than the '

                             'dimensionality of the target shape '

                             '(%d).' % (mtx.ndim, len(shape)))

        

        if exist_dims is not None:

            exist_dims = iterable(exist_dims)



            if len(exist_dims) != len(set(exist_dims)):

                raise ValueError('Existing dimensions should be unique.')

            

            if mtx.ndim != len(exist_dims):

                raise ValueError('Number of matrix dimensions (%d) '

                                 'should match the number of existing '

                                 'dimensions (%d).' % (mtx.ndim, len(exist_dims)))



            expand_dims = [i

                           for i in range(len(shape))

                           if i not in exist_dims]

                             

        elif expand_dims is not None:

            expand_dims = iterable(expand_dims)

            

            if len(expand_dims) != len(set(expand_dims)):

                raise ValueError('Expanding dimensions should be unique.')

            

            if len(shape) - mtx.ndim != len(expand_dims):

                raise ValueError('Dimensionality of the target shape '

                                 'minus the number of matrix dimensions '

                                 '(%d) should match the number of expanding '

                                 'dimensions (%d).' % (len(shape) - mtx.ndim, len(expand_dims)))

            

            exist_dims = [i

                          for i in range(len(shape))

                          if i not in expand_dims]

            

        else:

            raise ValueError('Target shape undetermined. Provide '

                             '``exist_dims`` or ``expand_dims``.')



        repeat = [n

                  for i, n in enumerate(shape)

                  if i in expand_dims]



        for i1, i2 in enumerate(exist_dims):

            if shape[i2] != mtx.shape[i1]:

                raise ValueError('Current matrix dimension (%d = %d) '

                                 'should match target shape (%d = %d).' % (i1, mtx.shape[i1], i2, shape[i2]))



    elif repeat is not None and expand_dims is not None:

        repeat = iterable(repeat)

        expand_dims = iterable(expand_dims)



        if len(expand_dims) != len(set(expand_dims)):

            raise ValueError('Expanding dimensions should be unique.')



        if len(repeat) != len(expand_dims):

            raise ValueError('Number of repititions (%d) should '

                             'match the number of expanding '

                             'dimensions (%d).' % (len(repeat), len(expand_dims)))

            

    else:

        raise ValueError('Target shape undetermined. Provide '

                         '``shape`` and ``exist_dims`` or '

                         '``expand_dims``, or ``repeat`` and ``expand_dims``.')



    for i, n in zip(expand_dims, repeat):

        mtx = np.expand_dims(mtx, i).repeat(n, axis=i)


    return mtx



def trapz_and_repeat(mtx, x, axis=-1):



    if axis < 0:

        axis += len(mtx.shape)



    return expand_and_repeat(np.trapz(mtx, x, axis=axis),
                             shape=mtx.shape, expand_dims=axis)

def spec2d(Hs, Tp, pdir, spread=30, units='deg', normalize=True):
    
    import numpy as np
    from oceanwaves.utils import trapz_and_repeat

    """ This function returns the 2d-spectrum from Hs, Tp, pdir and spread
    
    Edite by Nícolas and Marília to be used in the master project 

    -----------------------------------------------------

    REFERENCE:
    - https://www.orcina.com/webhelp/OrcaFlex/Content/html/Waves,Wavespectra.htm
    - http://research.dnv.com/hci/ocean/bk/c/a28/s3.htm
    - https://svn.oss.deltares.nl/repos/openearthtools/trunk/matlab/applications/waves/

    -----------------------------------------------------

    Parameters: 

    Hs = []; % Sig. higth 
    Tp = []; % Peak Period  
    pdir = [];  % Peak Direction  
    spread = []; % Spread in degrees. If no value is passed, default is used (30°)
    units = [deg] or [rad]; # Direction unit. Default is 'deg'
    
    

    
    
     """

    # Setup parameters:

    freq = np.array([0.0500, 0.0566, 0.0642, 0.0727, 0.0824, 0.0933, 0.1057, 0.1198,
                     0.1357, 0.1538, 0.1742, 0.1974, 0.2236, 0.2533, 0.2870, 0.3252,
                     0.3684, 0.4174, 0.4729, 0.5357, 0.6070, 0.6877, 0.7791, 0.8827,
                     1.0000]) # frequncy bin in swan

    directions =  np.arange(0., 360., 10) # Directional bin swan
    
    if Tp == 0:
        spec_2d = np.zeros((len(freq), len(directions)))
    else:
        fp = 1/Tp # peak frequency

        gam = 3.3 # default value according to SWAN manual

        g = 9.81 # gravity


        if spread > 31.5:
            ms = 1
        elif spread <= 31.5 and spread > 27.6:
            ms = 2
        elif spread <= 27.5 and spread > 24.9:
            ms = 3
        elif spread <= 24.9 and spread > 22.9:
            ms = 4
        elif spread <= 22.9 and spread > 21.2:
            ms = 5
        elif spread <= 21.2 and spread > 19.9:
            ms = 6
        elif spread <= 19.9 and spread > 18.8:
            ms = 7
        elif spread <= 18.8 and spread > 17.9:
            ms = 8
        elif spread <= 17.9 and spread > 17.1:
            ms = 9
        elif spread <= 17.1 and spread > 14.2:
            ms = 10
        elif spread <= 14.2 and spread > 12.4:
            ms = 15
        elif spread <= 12.4 and spread > 10.2:
            ms = 20
        elif spread <= 10.2 and spread > 8.9:
            ms = 30
        elif spread <= 8.9 and spread > 8:
            ms = 40
        elif spread <= 8 and spread > 7.3:
            ms = 50
        elif spread <= 7.3 and spread > 6.8:
            ms = 60
        elif spread <= 6.8 and spread > 6.4:
            ms = 70
        elif spread <= 6.4 and spread > 6.0:
            ms = 80
        elif spread <= 6.0 and spread > 5.7:
            ms = 90
        elif spread <= 5.7 and spread > 4:
            ms = 100
        elif spread <= 4 and spread > 2.9:
            ms = 400
        elif spread <= 7.3 and spread >= 2.0:
            ms = 800
        else:
            ms = 1000

        sigma = freq * 0
        sigma[freq<fp] = 0.07
        sigma[freq>=fp]= 0.09
        sigma = np.array(sigma)

        # Pierson-Moskowitz Spectrum

        alpha = 1 / (0.06533 * gam ** 0.8015 + 0.13467)/16; # Yamaguchi (1984), used in SWAN

        pm = alpha * Hs ** 2 * Tp **-4 * freq ** -5 * np.exp(-1.25 * (Tp * freq)**-4) 

        # apply JONSWAP shape

        jon = pm * gam ** np.exp(-0.5 * (Tp * freq - 1) ** 2. / (sigma ** 2.))

        jon[np.isnan(jon)] = 0

        # Optionally correct total energy of user-discretized spectrum to match Hm0, 
        # as above methods are only an approximation

        eps = np.finfo(float).eps

        if normalize is True:
            corr = Hs ** 2/(16*trapz_and_repeat(jon, freq))
            jon = jon*corr


        # Directional Spreading

        '''Generate wave spreading'''

        from math import gamma

        directions = np.asarray(directions, dtype=np.float)

        # convert units to radians
        if units.lower().startswith('deg'):
            directions = np.radians(directions)
            pdir = np.radians(pdir)
        elif units.lower().startswith('rad'):
            pass
        else:
            raise ValueError('Unknown units: %s')

        # compute directional spreading
        A1 = (2.**ms) * (gamma(ms / 2 + 1))**2. / (np.pi * gamma(ms + 1))
        cdir = A1 * np.maximum(0., np.cos(directions - pdir))
        #cdir = np.maximum(0., np.cos(directions - pdir))**s

        # convert to original units
        if units.lower().startswith('deg'):
            directions = np.degrees(directions)
            pdir = np.degrees(pdir)
            cdir = np.degrees(cdir)

        # normalize directional spreading
        if normalize:
            cdir /= trapz_and_repeat(cdir, directions - pdir, axis=-1)

        cdir = np.array([list(cdir)] * len(jon))

        jon_list = list(jon)
        
        jon = np.array([ele for ele in jon_list for i in range(len(directions))]
                      ).reshape(len(freq), len(directions))

        jon2 = jon * cdir
        jon2[np.isnan(jon2)] = 0
        spec_2d = jon2
        
    
    return spec_2d

def spec_points_satellite(coords, part_list, year, dataset_dirs, vars2drop):
    

    ''' #### WHAT ####
       - Calculate specs from a vector of coordinates and a xarray object.
       
       INPUT:
       - coords: array/list of coordinates
       - part_list: list of names of variables as in the xarray object (hs, tp, dp, ms)
       - wv_xr: dataset containing hs, tp, dp, ms

    '''

    import xarray as xr
    import numpy as np
    import pandas as pd
    from spec2d_era import spec2d
    import glob
    
    # Select desired sites from ERA netcdf
    wv_points = []
    for ds in dataset_dirs:
        file = glob.glob(f'{ds}/*.nc')

        ds = xr.open_dataset(file[0], drop_variables=vars2drop)

        ds = ds.assign_coords(time=pd.date_range(start='1979-01-01', 
                                                 end='2018-10-31T23:00:00', 
                                                 freq='H'))
        ds = ds.sel(time=slice(f'{year}-01-01T00:00:00', f'{year}-12-31T23:00:00'))

        wv_points.append(ds)

    # Create dataframe from xr datasets stored in wv_points
    dfs = []
    
    for index, ds in enumerate(wv_points):
        df = wv_points[index]
        df = pd.DataFrame(dict(hs=df[part_list[0]].values,
                               tp=df[part_list[1]].values,
                               dp=df[part_list[2]].values,
                               ms=df[part_list[3]].values
                              )
                         )
        dfs.append(df)
        
    # Calculate specs from data in dfs
    specs = []

    for df in dfs:
        loc = []
        for row in df.itertuples():
            spec = spec2d(row.hs, row.tp, row.dp, row.ms)
            loc.append(spec)
        specs.append(loc)

    specs = np.array(specs)
    print(specs.shape)
    
    return specs


def spec_points(coords, part_list, wv_xr):
    

    ''' #### WHAT ####
       - Calculate specs from a vector of coordinates and a xarray object.
       
       INPUT:
       - coords: array/list of coordinates
       - part_list: list of names of variables as in the xarray object (hs, tp, dp, ms)
       - wv_xr: dataset containing hs, tp, dp, ms

    '''

    import xarray as xr
    import numpy as np
    import pandas as pd
    from spec2d_era import spec2d
    
    # Select desired sites from ERA netcdf
    wv_points = []

    for coord in coords:
        wv_point = wv_xr.sel(dict(latitude=coord[1],
                                  longitude=coord[0]),
                            method='nearest')
        wv_points.append(wv_point)
    

    # Create dataframe from xr datasets stored in wv_points
    dfs = []
    
    for index, ds in enumerate(wv_points):
        df = wv_points[index]
        df = pd.DataFrame(dict(hs=df[part_list[0]].values,
                               tp=df[part_list[1]].values,
                               dp=df[part_list[2]].values,
                               ms=df[part_list[3]].values
                              )
                         )
        dfs.append(df)
        
    # Calculate specs from data in dfs
    specs = []

    for df in dfs:
        loc = []
        for row in df.itertuples():
            spec = spec2d(row.hs, row.tp, row.dp, row.ms)
            loc.append(spec)
        specs.append(loc)

    specs = np.array(specs)
    print(specs.shape)
    
    return specs

def spec_points(coords, part_list, wv_xr):
    

    ''' #### WHAT ####
       - Calculate specs from a vector of coordinates and a xarray object.
       
       INPUT:
       - coords: array/list of coordinates
       - part_list: list of names of variables as in the xarray object (hs, tp, dp, ms)
       - wv_xr: dataset containing hs, tp, dp, ms

    '''

    import xarray as xr
    import numpy as np
    import pandas as pd
    from spec2d_era import spec2d
    
    # Select desired sites from ERA netcdf
    wv_points = []

    for coord in coords:
        wv_point = wv_xr.sel(dict(latitude=coord[1],
                                  longitude=coord[0]),
                            method='nearest')
        wv_points.append(wv_point)
    

    # Create dataframe from xr datasets stored in wv_points
    dfs = []
    
    for index, ds in enumerate(wv_points):
        df = wv_points[index]
        df = pd.DataFrame(dict(hs=df[part_list[0]].values,
                               tp=df[part_list[1]].values,
                               dp=df[part_list[2]].values,
                               ms=df[part_list[3]].values
                              )
                         )
        dfs.append(df)
        
    # Calculate specs from data in dfs
    specs = []

    for df in dfs:
        loc = []
        for row in df.itertuples():
            spec = spec2d(row.hs, row.tp, row.dp, row.ms)
            loc.append(spec)
        specs.append(loc)

    specs = np.array(specs)
    print(specs.shape)
    
    return specs

def write_specs(time, coords, specs, output_filename, ext='.bnd'):
    
    import pandas as pd
    
    ''' 
    WHAT: 
    
    Writes Swan spectral file from spec numpy array
    
    INPUT:
    time: time vector with same length as second dimension os specs
    coords: list of lists of [[lon1, lat1], [lon2, lat2], ...]. same 
    length as first dimension of specs
    specs: 4d numpy array with dims [location, time, freq, dir]
    output_filename: str to be inside the filename 
    ext: str with the extension of the file. default option is '.bnd'
    '''
   
    datetime = pd.to_datetime(time)

    for loc, coord in enumerate(coords):
        fin = open('specs_' + output_filename + str(loc+1) + str(ext), 'w')
        fin.write(
    f'''SWAN   1                                Swan standard spectral file, version
$   Data produced by SWAN version 40.51AB             
$   Project:                 ;  run number:     
TIME                                    time-dependent data
     1                                  time coding option
LONLAT                                  locations in spherical coordinates
1
     {coord[0]:.6f}      {coord[1]:.6f}
AFREQ                                   absolute frequencies in Hz
    25
         0.0500
         0.0566
         0.0642
         0.0727
         0.0824
         0.0933
         0.1057
         0.1198
         0.1357
         0.1538
         0.1742
         0.1974
         0.2236
         0.2533
         0.2870
         0.3252
         0.3684
         0.4174
         0.4729
         0.5357
         0.6070
         0.6877
         0.7791
         0.8827
         1.0000
NDIR                                   spectral nautical directions in degr
36
       0
       10
       20
       30
       40
       50
       60
       70
       80
       90
       100
       110
       120
       130
       140
       150
       160
       170
       180
       190
       200
       210
       220
       230
       240
       250
       260
       270
       280
       290
       300
       310
       320
       330
       340
       350
QUANT
     1                                  number of quantities in table
VaDens                                  variance densities in m2/Hz/degr
m2/Hz/degr                              unit
   -0.9999                              exception value
''')
        fin.close()
        fin=open('specs_' + output_filename + str(loc+1) + str(ext), "a+")
        for line in range(len(datetime)):
            fin.write(
    '''{:%Y%m%d.%H%M}
FACTOR
1
{:>10}'''.format(
        datetime[line],
        pd.DataFrame(specs[loc][line][:][:]).to_csv(index=False,
                                                   header=False,
                                                   sep=',',
                                                   float_format='%7.5f',
                                                   na_rep= -0.9999,
                                                   line_terminator='\n')))
        fin = open('specs_' + output_filename + str(loc+1) + str(ext), "rt")
        data = fin.read()
        data = data.replace(',', ' ')
        fin.close()

        fin = open('specs_' + output_filename + str(loc+1) + str(ext), "wt")
        fin.write(data)
        fin.close()

