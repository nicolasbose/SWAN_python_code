def plot_validation(data, palette, lim, step, unit, alpha=0.08):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    sns.set_context('poster') 
    

    g = sns.lmplot(x='Buoy', y='value', 
                   data=data, 
                   row='Loc', col='Wind forcing', 
                   hue='wave', palette=palette,
                   scatter_kws=dict(edgecolor="k", 
                                    linewidth=0.5,
                                    alpha=alpha),
                   hue_order=['ERA5'],
                   legend=False,
                   height=9.44)

    g.set_axis_labels(f"Buoy ({unit})", f"SWAN ({unit})").set(
        xlim=(0, lim), ylim=(0, lim),
        xticks=np.arange(0, (lim + step), step), 
        yticks=np.arange(0, (lim + step), step)).fig.subplots_adjust(wspace=.08)
   
    ax = g.axes

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='o', markerfacecolor=palette[1], 
                              label='ERA5', color='w', markersize=15)]

    leg = ax[2, 1].legend(handles=legend_elements,
                          title=r'$\bf{Boundary}$'' 'r'$\bf{condition}$', 
                          loc='lower center', 
                          bbox_to_anchor=(0.52, -0.35),
                          ncol=3,
                          labels=['ERA5'],
                          fancybox=True, framealpha=1, 
                          shadow=False, borderpad=1)
    leg._legend_box.align='center'

    plt.show()

    return g



def plot_timeseries(data, df_buoy, palette, lim, step, unit):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    g = sns.FacetGrid(data,
                      row='Loc', col='Wind forcing', 
                      hue='Boundary condition',
                      aspect=1.6,
                      palette=palette)
    ax = g.axes

    for i in range(0,3):
        ax[0, i].scatter(df_buoy[(df_buoy['Boundary condition'] == 'Buoy') &
                        (df_buoy['Loc'] == 'Itajaí')]['Date'],
                 df_buoy[(df_buoy['Boundary condition'] == 'Buoy') &
                        (df_buoy['Loc'] == 'Itajaí')]['value'],
                 color='k', s=1, marker='o')

    for i in range(0,3):
        ax[1, i].scatter(df_buoy[(df_buoy['Boundary condition'] == 'Buoy') &
                        (df_buoy['Loc'] == 'Tramandaí')]['Date'],
                 df_buoy[(df_buoy['Boundary condition'] == 'Buoy') &
                        (df_buoy['Loc'] == 'Tramandaí')]['value'],
                 color='k', s=1, marker='o')

    for i in range(0,3):
        ax[2, i].scatter(df_buoy[(df_buoy['Boundary condition'] == 'Buoy') &
                        (df_buoy['Loc'] == 'Rio Grande')]['Date'],
                 df_buoy[(df_buoy['Boundary condition'] == 'Buoy') &
                        (df_buoy['Loc'] == 'Rio Grande')]['value'],
                 color='k', s=1, marker='o')


    g.map(plt.plot, 'Date', 'value', linewidth=1, alpha=0.8)    

    g.set_axis_labels("", f"SWAN ({unit})").set(xlim=(pd.to_datetime('2015-01-01'),
                                                        pd.to_datetime('2015-12-31')), 
                                              ylim=(0, lim),
                                              xticks=pd.date_range(start='2015-01',
                                                                   end='2016-01',
                                                                   freq='M'),
                                              xticklabels=pd.date_range(start='2015-01', 
                                                                        end='2016-01', 
                                                                        freq='M').strftime('%b'),
                                              yticks=np.arange(0, (lim + step), 
                                                               step)).fig.subplots_adjust(wspace=.08)

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='o', markerfacecolor='k', 
                              label='Buoy', color='w', markersize=6),
                       Line2D([0], [0], color=palette[1], 
                              label='CAWCAR'),
                       Line2D([0], [0], color=palette[0], 
                              label='ERA5'),
                       Line2D([0], [0], color=palette[2], 
                              label='WW3')]

    leg = ax[2, 1].legend(handles=legend_elements,
                          title=r'$\bf{Boundary}$'' 'r'$\bf{condition}$', 
                          loc='lower center', 
                          bbox_to_anchor=(0.52, -0.55),
                          ncol=4,
                          labels=['Buoy','CAWCAR', 'ERA5', 'WW3'],
                          fancybox=True, framealpha=1, 
                          shadow=False, borderpad=1)
    leg._legend_box.align='center'

    plt.show()
    
    return g


def prepare2plot(files):
    
    ''' Read, merge and organise data to plot
    
    INPUT:
    CSIRO, ERA and WW3 files containing buoy,
    csiro, era and gfs outputs.
    
    '''
    
    
    import pandas as pd

    df_csiro = pd.read_csv(files[0], parse_dates=['date'])
    df_era = pd.read_csv(files[1],  usecols=['date','era', 'csiro', 'gfs'],
                         parse_dates=['date'])
    df_ww3 = pd.read_csv(files[2],  usecols=['date','era', 'csiro', 'gfs'],
                    parse_dates=['date'])
    df = df_csiro.merge(df_era, on='date', how='inner', 
                         suffixes=('_csiro', ''))
    df = df.merge(df_ww3, on='date', how='inner', 
                   suffixes=('_era', '_ww3'))


    columns = ['Buoy', 'ERA5', 'CFSR', 'GFS']
    df_csiro = df[['buoy', 'era_csiro',
                   'csiro_csiro', 'gfs_csiro']]
    df_csiro.columns = columns
    df_era = df[['buoy', 'era_era',
                 'csiro_era', 'gfs_era']]
    df_era.columns = columns
    df_ww3 = df[['buoy', 'era_ww3',
                 'csiro_ww3', 'gfs_ww3']]
    df_ww3.columns = columns

    dfs = [df_csiro, df_era, df_ww3]
    wave = ['CSIRO', 'ERA5', 'WW3']
    dfs_scatter_melt = []

    for index, dataframe in enumerate(dfs):
        df_scatter_molten = pd.melt(dataframe, id_vars='Buoy',
                            var_name='Wind forcing')
        df_scatter_molten['wave'] = wave[index]
        dfs_scatter_melt.append(df_scatter_molten)
        
    dfs_scatter_molten = pd.concat(dfs_scatter_melt)
        
    columns = ['Date', 'Buoy', 'ERA5', 'CSIRO', 'WW3']
    df_csiro = df[['date', 'buoy', 'csiro_era',
                   'csiro_csiro', 'csiro_ww3']]
    df_csiro.columns = columns
    df_era = df[['date', 'buoy', 'era_era',
                 'era_csiro', 'era_ww3']]
    df_era.columns = columns
    df_gfs = df[['date', 'buoy', 'gfs_era',
                 'gfs_csiro', 'gfs_ww3']]
    df_gfs.columns = columns

    dfs = [df_csiro, df_era, df_gfs]
    wind = ['ERA5', 'CFSR', 'GFS']
    dfs_ts_melt = []

    for index, dataframe in enumerate(dfs):
        df_ts_molten = pd.melt(dataframe, id_vars='Date',
                            var_name='Boundary condition')
        df_ts_molten['Wind forcing'] = wind[index]
        dfs_ts_melt.append(df_ts_molten)
    

    dfs_ts_molten = pd.concat(dfs_ts_melt)

    return df, dfs_scatter_molten, dfs_ts_molten

def validation_stats_ro(df, location, parameter):
    
    import warnings
    warnings.simplefilter('ignore')
   
    import numpy as np
    import pandas as pd
    
    def sse(x, y):
        return np.mean((np.mean(x) - y) **2)

    def rmse(column):
        from sklearn.metrics import mean_squared_error
        return (mean_squared_error(df['buoy'], column) ** 0.5)

    def SI(column):
        return rmse(column)/np.mean(column)

    def bias(column):
        return sse(column, df['buoy']) - np.var(column)

    def error(column):
        return bias(column)**2 + np.var(column) + np.std(column)**2

    stats_df = df[['buoy', 'era' 
                   ]].apply([np.mean, SI, rmse])
    bias_list = []
    for column in df[['buoy', 'era']]:
        bias_list.append(bias(df[column]))

    stats_df = stats_df.append(pd.DataFrame([bias_list], 
                                              columns=stats_df.columns, 
                                              index=['bias']))
    from scipy.stats import pearsonr

    corr_list = []
    for column in df[['buoy', 'era']]:
        corr = pearsonr(df['buoy'], df[column])
        corr_list.append(corr[0])
        
    stats_df = stats_df.append(pd.DataFrame([corr_list], 
                                          columns=stats_df.columns, 
                                          index=['Corr']))

        
    stats_df.index = ['Mean', 'SI', 'RMSE',
                      'Bias', 'Corr']

    stats_melt = pd.melt(frame=stats_df.reset_index(),
                             id_vars='index', var_name='reanalysis')
    stats_melt['par'] = parameter
    stats_melt['location'] = location
    stats_melt.columns = ['stats', 'reanalysis', 'value', 'parameter', 'location']

    
    return stats_df, stats_melt

def validation_stats(df, location, parameter):
    
    import warnings
    warnings.simplefilter('ignore')
   
    import numpy as np
    import pandas as pd
    
    def sse(x, y):
        return np.mean((np.mean(x) - y) **2)

    def correl(column):
        corr = np.corrcoef(df['buoy'], column)
        return corr[0][1]

    def rmse(column):
        from sklearn.metrics import mean_squared_error
        return (mean_squared_error(df['buoy'], column) ** 0.5)

    def SI(column):
        return rmse(column)/np.mean(column)

    def bias(column):
        return sse(column, df['buoy']) - np.var(column)

    def error(column):
        return bias(column)**2 + np.var(column) + np.std(column)**2

    stats_df = df[['buoy', 'era']].apply([np.mean, SI, 
                                           correl, rmse])
    bias_list = []
    for column in df[['buoy', 'era']]:
        bias_list.append(bias(df[column]))

    stats_df = stats_df.append(pd.DataFrame([bias_list], 
                                              columns=stats_df.columns, 
                                              index=['bias']))

        
    stats_df.index = ['Mean', 'SI', 'Corr', 
                      'RMSE', 'Bias']

    stats_melt = pd.melt(frame=stats_df.reset_index(),
                             id_vars='index', var_name='reanalysis')
    stats_melt['par'] = parameter
    stats_melt['location'] = location
    stats_melt.columns = ['stats', 'reanalysis', 'value', 'parameter', 'location']

    
    return stats_df, stats_melt

def read_buoy_csv(file, par, par_index, sep=';'):
    
    import pandas as pd
    import warnings
    warnings.simplefilter('ignore')

    parser = lambda x: pd.datetime.strptime(x, '%Y.0 %m.0 %d.0 %H.0')
    csv = pd.read_csv(file, skiprows=0, sep=sep, header=0, 
                      usecols=[3, 4, 5, 6, 7, 8, par_index+1], 
                      parse_dates={'date': [2,3,4,5]},
                      date_parser=parser)
    csv.columns = ['date', 'lat', 'lon', par]
    csv = csv.set_index('date')

    return csv

def mat2list(mat_file, grid_mat, lon, lat, par_list):
    
    import warnings
    warnings.simplefilter('ignore')
        
    # Importa output para um dicionário
    import scipy.io as sio
    mat = sio.loadmat(mat_file) # se quiser checar quais variáveis estão presentes, digita mat.keys() em uma nova célula
    grid = sio.loadmat(grid_mat)
    drop = ['__header__', '__version__', '__globals__'] # variáveis que não quero

    keys = [key for key in mat.keys() if key not in drop] # seleciono as variáveis que quero, ou seja, que não estão em drop
    lons = grid['Xp'] # seleciono longitude como a variável 'Xp' de grid
    lats = grid['Yp'] # seleciono latitude como a variável 'Yp' de grid

    import numpy as np
    
    dif_lons = abs(lons - lon)
    dif_lats = abs(lats - lat)
    coords = np.where((dif_lons*dif_lats) == (dif_lons * dif_lats).min())
      
    for key in keys:
        par_list.append(mat[key][coords[0][0]][coords[1][0]])
        
    return keys

def list2df(mat_list, par_name, lon, lat, dt_format):
    
    import warnings
    warnings.simplefilter('ignore')
    import pandas as pd
    import manuscript_functions as mf
    
    par = []
    timestamps = []
    
    for mat_file in mat_list:
        timestamps.append(mf.mat2list(mat_file, 'grid.mat', 
                                      lon, lat, par))

    df = pd.DataFrame({'date': [item for sublist in 
                                timestamps for item in 
                                sublist],
                       par_name: par})

    df['date'] = pd.to_datetime(df['date'], format=dt_format)
    df = df.set_index('date')
    
    return df

def knn_filter(buoy_filepath, output_filepath, cols, pnboia=True):    

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly
    import plotly_express as px
    import plotly.graph_objects as go
    import plotly.offline as py

    import seaborn as sns

    from numpy import percentile
    from pyod.models.cblof import CBLOF
    from pyod.models.iforest import IForest
    from pyod.models.hbos import HBOS
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.cof import COF
    from plotly.subplots import make_subplots 
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler

    mpl.rcParams["figure.dpi"] = 300

    if pnboia is True:

        wave = pd.read_csv(buoy_filepath, sep=';')
        wave[wave[cols[0]] == -9999] = np.nan
        wave[wave[cols[1]] == -9999] = np.nan
        wave[wave['Wvhtflag'] != 0] = np.nan
        wave[wave['Dpdflag'] != 0] = np.nan
        wave = wave.dropna()
        raw = pd.read_csv(buoy_filepath, sep=';')
        raw[raw[cols[0]] == -9999] = np.nan
        raw[raw[cols[1]] == -9999] = np.nan
        raw[raw['Wvhtflag'] != 0] = np.nan
        raw[raw['Dpdflag'] != 0] = np.nan
        raw = raw.dropna()
    else:
        import glob
        files = glob.glob(buoy_filepath)
        wave = pd.concat(pd.read_csv(f, sep='\t', skiprows=0, 
                                     header=0, engine='python') for f in files)
        wave = wave.dropna()
        raw = pd.concat(pd.read_csv(f, sep='\t', skiprows=0, 
                                    header=0, engine='python') for f in files)
        raw = raw.dropna()

    minmax = MinMaxScaler(feature_range=(0, 1))
    wave[cols] = minmax.fit_transform(wave[cols])
    wave[cols].head()

    X1 = wave[cols[1]].values.reshape(-1, 1)
    X2 = wave[cols[0]].values.reshape(-1, 1)

    X = np.concatenate((X1, X2), axis=1)

    outliers_fraction = 0.01

    classifiers = {"K Nearest Neighbors (KNN)": KNN(contamination=outliers_fraction)}

    xx , yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    outliers = {}

    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)

        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1

        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)
        plt.figure(figsize=(7, 7))

        # copy of dataframe
        df = wave.copy()
        df['outlier'] = y_pred.tolist()

        # creating a combined dataframe of outliers from the 4 models
        outliers[clf_name] = df.loc[df['outlier'] == 1]

        # IN1 - inlier feature 1,  IN2 - inlier feature 2
        IN1 =  np.array(df[cols[1]][df['outlier'] == 0]).reshape(-1,1)
        IN2 =  np.array(df[cols[0]][df['outlier'] == 0]).reshape(-1,1)


        # OUT1 - outlier feature 1, OUT2 - outlier feature 2
        OUT1 =  df[cols[1]][df['outlier'] == 1].values.reshape(-1,1)
        OUT2 =  df[cols[0]][df['outlier'] == 1].values.reshape(-1,1)

        print('OUTLIERS:',n_outliers, '|', 'INLIERS:',n_inliers, '|', 'MODEL:',clf_name)

            # threshold value to consider a datapoint inlier or outlier
        threshold = stats.scoreatpercentile(scores_pred,100 * outliers_fraction)

        # decision function calculates the raw anomaly score for every point
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        Z = Z.reshape(xx.shape)

        # fill blue map colormap from minimum anomaly score to threshold value
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.GnBu_r)

        # draw red contour line where anomaly score is equal to thresold
        a = plt.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')

        # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
        plt.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='lemonchiffon')

        b = plt.scatter(IN1,IN2, c='white',s=20, edgecolor='k')

        c = plt.scatter(OUT1,OUT2, c='black',s=20, edgecolor='k')

        plt.axis('tight')  

        # loc=2 is used for the top left corner 
        plt.legend(
            [a.collections[0], b,c],
            ['Decision function', 'Inliers','Outliers'],
            prop=mpl.font_manager.FontProperties(size=13),
            loc=2)

        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=np.round(np.linspace(round(raw[cols[0]].min()), 
                                        round(raw[cols[0]].max()), 
                                        6), 1))
        plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    labels=np.round(np.linspace(round(raw[cols[1]].min()), 
                                        round(raw[cols[1]].max()), 
                                        6), 1))
        plt.xlabel('Tp (s)')
        plt.ylabel('Hs (m)')
        plt.title(clf_name)
        plt.show()

        raw['outlier'] = df['outlier']
        filtered = raw[raw['outlier'] != 1]

        filtered.to_csv(output_filepath, sep=';')
        