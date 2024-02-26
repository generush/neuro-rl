# https://plotly.com/python/3d-scatter-plots/

from typing import List

import numpy as np

import matplotlib
# matplotlib.use('TkAgg')  # Replace 'TkAgg' with another backend if needed
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib.cm as cm

def plot_data(x_data, y_data, z_data, c_data, cc_global_min, cc_global_max, data_type, zlabel, clabel, cmap, path, save_figs = False):
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = x_data
    yy = y_data
    zz = z_data
    cc = c_data

    # plot figures with speed colors and tangling colors
    ax.scatter(xx, yy, zz, c=cc, s=2*np.ones_like(xx), cmap=cmap, vmin=cc_global_min, vmax=cc_global_max, alpha=1, depthshade=True, rasterized=True)
    ax.scatter(xx.flatten(), yy.flatten(), 1.5 * zz.min()*np.ones_like(zz), c='grey', s=1*np.ones_like(xx), alpha=1, depthshade=True, rasterized=True)

    # Add labels and a legend
    xlabel = 'PC 1'
    ylabel = 'PC 2'
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel(zlabel)
    
    current_backend = matplotlib.get_backend()
    if current_backend != 'module://ipympl.backend_nbagg':

        manager = plt.get_current_fig_manager()
        manager.window.title(data_type + ' /// ' + xlabel + ylabel + zlabel + ' /// ' + clabel + ' /// ' + ' interpolated average cycles')


    print('----------------------------------------------------------------------------------------------------------')
    print(data_type + ' /// ' + xlabel + ylabel + zlabel + ' /// ' + clabel + ' /// ' + ' interpolated average cycles')

    ax.view_init(20, 55)

    norm = mcolors.Normalize(vmin=cc_global_min, vmax=cc_global_max)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75)
    cbar.set_label(clabel, labelpad=-29, y=1.1, rotation=0)

    cax = cbar.ax
    cax.set_position([0.80, 0.15, 0.02, 0.5])

    plt.show()

    if save_figs:
        filename = f"{data_type}__{xlabel}{ylabel}{zlabel}__{clabel}".replace("/", "-")
        fig.savefig(path + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())

def plot_pc12_speed_axis(df, data_names):
        # export some data for the paper/suppl matl
        # pd.DataFrame(pc_12speed_tf).to_csv(path + 'info_' + data_type + '_pc_12speed_tf.csv')
        # pd.DataFrame(pc_123_tf).to_csv(path + 'info_' + data_type + '_pc_123_tf.csv')
        # pd.DataFrame(pca.explained_variance_ratio_.cumsum()).to_csv(path + 'info_' + data_type + '_cumvar.csv')
        # pd.DataFrame(x).to_csv(path + 'info_' + data_type + '_x_by_speed.csv')
        # pd.DataFrame(y).to_csv(path + 'info_' + data_type + '_y_by_speed.csv')
        # pd.DataFrame(z1).to_csv(path + 'info_' + data_type + '_z1_by_speed.csv')
        # pd.DataFrame(z2).to_csv(path + 'info_' + data_type + '_z2_by_speed.csv')
        # pd.DataFrame(t).to_csv(path + 'info_' + data_type + '_tangling_by_speed.csv')
        # pd.DataFrame(d_opt).to_csv(path + 'info_' + data_type + '_dopt.csv')
        # export_scl(scl, path + 'info_' + data_type +'_SCL' + '.pkl')
        # export_pca(pca, path + 'info_' + data_type +'_PCA' + '.pkl')
 
    s_global_min = df.loc[:, df.columns.str.contains('OBS_RAW_009_u_star')].values.min()
    s_global_max = df.loc[:, df.columns.str.contains('OBS_RAW_009_u_star')].values.max()
    t_global_min = df.loc[:, df.columns.str.contains('TANGLING')].values.min()
    t_global_max = df.loc[:, df.columns.str.contains('TANGLING')].values.max()
 
    # pd.DataFrame(pca.explained_variance_ratio_.cumsum()).to_csv(data_type + '_cumvar.csv')
 
    # Plot the data for each data_type
    for idx, data_type in enumerate(data_names):
        speed = df.loc[:, df.columns.str.contains('OBS_RAW_009_u_star')].values
        pc1 = df.loc[:, df.columns.str.contains(data_type + '_PC_001')].values
        pc2 = df.loc[:, df.columns.str.contains(data_type + '_PC_002')].values
        pc3 = df.loc[:, df.columns.str.contains(data_type + '_PC_003')].values
        speed_axis = df.loc[:, df.columns.str.contains(data_type + '_SPEED_AXIS')].values
        tangling = df.loc[:, df.columns.str.contains(data_type + '_TANGLING')].values

        
        path = ""
        # PC1, PC2, PC3, Speed
        plot_data(pc1, pc2, pc3, speed, s_global_min, s_global_max, data_type, 'PC 3', 'u [m/s]', 'Spectral', path, save_figs=True)

        # PC1, PC2, PC3, Tangling
        plot_data(pc1, pc2, pc3, tangling, np.min(tangling), np.max(tangling), data_type, 'PC 3', 'Tangling', 'viridis', path, save_figs=True)

        # PC1, PC2, SpeedAxis, Speed
        plot_data(pc1, pc2, speed_axis, speed, s_global_min, s_global_max, data_type, 'Speed Axis', 'u [m/s]', 'Spectral', path, save_figs=True)

        # PC1, PC2, SpeedAxis, Tangling
        plot_data(pc1, pc2, speed_axis, tangling, np.min(tangling), np.max(tangling), data_type, 'Speed Axis', 'Tangling', 'viridis', path, save_figs=True)


    # crop white space out of pdfs
    # crop_pdfs_in_folder(path)
    
    print('done plotting')
    

