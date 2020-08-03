import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import pandas as pd

import yt
  

# Note! sgals = list of series of the df!
# examples
# sgals = [sgal_R_df.loc[tested_sgal_sid]]
# sgals = [sgal_R_df.loc[i] for i in sgal_R_df.iloc[[1,13,14]].index]
def plot_galaxies(description, sim_line, sgals_lines, plt_range, Rvir, center=None,ds=None, \
                traced_stars_indices=[], traced_dm_indices=[], traced_gas_indices = [], sgal_rs=None, show=True,\
                vmin=3, vmax=9):    
    if (None == ds):
        if (not sim_line.empty):
            ds = yt.load(sim_line['sim_path'])
        else:
            raise ValueError('No value at both ds and sim_line')    
    if (None == center):
#        # Check if sgals is single sgal or more
#        if isinstance(sgals[0], pd.core.series.Series):
#            center = sgals[0][['zana_x','zana_y', 'zana_z']].tolist()
#        else:
        raise ValueError('invalid value in center')
    #center = [0.,0.,0.]
    # Gather Particles data:
    center_pos = ds.arr(center, 'code_length')
    range_vec = ds.arr(plt_range * np.ones(3), 'kpc').in_units('code_length') 
    region_galaxy = ds.region(center_pos, center_pos-range_vec, center_pos+range_vec)
    star_particles_x    = (region_galaxy[ ('stars', 'particle_position_x')]-center_pos[0]).in_units('kpc')
    star_particles_y    = (region_galaxy[ ('stars', 'particle_position_y')]-center_pos[1]).in_units('kpc')
    star_particles_z    = (region_galaxy[ ('stars', 'particle_position_z')]-center_pos[2]).in_units('kpc')
    star_particles_mass = (region_galaxy[ ('stars', 'particle_mass')]).in_units('Msun')
    dm_particles_x      = (region_galaxy[ ('darkmatter', 'particle_position_x')]-center_pos[0]).in_units('kpc')
    dm_particles_y      = (region_galaxy[ ('darkmatter', 'particle_position_y')]-center_pos[1]).in_units('kpc')
    dm_particles_z      = (region_galaxy[ ('darkmatter', 'particle_position_z')]-center_pos[2]).in_units('kpc')
    dm_particles_mass   = (region_galaxy[ ('darkmatter', 'particle_mass')]).in_units('Msun')
    gas_particles_x     = (region_galaxy[ ('index', 'x')]-center_pos[0]).in_units('kpc')
    gas_particles_y     = (region_galaxy[ ('index', 'y')]-center_pos[1]).in_units('kpc')
    gas_particles_z     = (region_galaxy[ ('index', 'z')]-center_pos[2]).in_units('kpc')
    gas_particles_mass  = (region_galaxy[ ('gas', 'cell_mass')]).in_units('Msun')
    
    # Gather Galaxies data:
    def gal_cen_in_region(g):
        if range_vec[0] < abs(ds.quan(g['zana_x'],'code_length')-center_pos[0]) :
            return False
        if range_vec[1] < abs(ds.quan(g['zana_y'],'code_length')-center_pos[1]) :
            return False
        if range_vec[2] < abs(ds.quan(g['zana_z'],'code_length')-center_pos[2]) :
            return False
        return True
    
    #sgals_lines = filter(gal_cen_in_region, sgals_lines) # TODO FIXME
    ###original line: g_x   = map(lambda sgid:((ds.quan(sgals_lines.loc[sgid]['zana_x'],'code_length')-center_pos[0]).in_units('kpc').v), sgals_lines.index)
    if len(sgals_lines.index)>=6:
        g_x   = map(lambda sgid:((ds.quan(sgals_lines.loc['zana_x'],'code_length')-center_pos[0]).in_units('kpc').v),[0])
        g_y   = map(lambda sgid:((ds.quan(sgals_lines.loc['zana_y'],'code_length')-center_pos[1]).in_units('kpc').v),[0])
        g_z   = map(lambda sgid:((ds.quan(sgals_lines.loc['zana_z'],'code_length')-center_pos[2]).in_units('kpc').v),[0])
    else:
        g_x   = map(lambda sgid:((ds.quan(sgals_lines.loc['zana_x'],'code_length')-center_pos[0]).in_units('kpc').v), sgals_lines.index)
        g_y   = map(lambda sgid:((ds.quan(sgals_lines.loc['zana_y'],'code_length')-center_pos[1]).in_units('kpc').v), sgals_lines.index)
        g_z   = map(lambda sgid:((ds.quan(sgals_lines.loc['zana_z'],'code_length')-center_pos[2]).in_units('kpc').v), sgals_lines.index)
    
#    ds.arr(sgals_lines.loc[sgals_lines.index]['zana_x'].values, 'code_length')
    if (None != sgal_rs):
         g_r = ds.arr(sgal_rs, 'kpc').v
    else:
        g_r   = ds.arr(sgals_lines.loc[sgals_lines.index]['Rsat[kpc]'].values,'kpc').v

    g_m   = ds.arr(sgals_lines.loc[sgals_lines.index]['Mtot(Reff)'],'Msun').v
    
    g_id  = sgals_lines.loc[sgals_lines.index]['zana_gid']#.values 
    sgals_param = (g_x, g_y, g_z, g_r, g_m, g_id, )

    print 'sgals_param:', sgals_param,'\n', len(sgals_param)
    

    title_star  = description + "Star mass projection - " + sim_line.name
    title_dm    = description + "DM mass projection - "   + sim_line.name
    title_gas   = description + "Gas mass projection - "  + sim_line.name
    title_total = description + "Total mass projection - "  + sim_line.name
      
    
    # trace particles:
    if [] != traced_stars_indices:
        dd=ds.all_data()
#        Note! Fortran to YT translation!!!!
#        cur_stars_list = list(cur_stars)
#         cur_stars_list.sort()
#         # subtracting 1 because fortran starts indexing from 1 instead of 0, go figure... :$
#         c_stars = np.array(cur_stars_list)-1 
#         # mask = np.in1d(dd[('stars', 'particle_index')], c_stars) # + ds.parameters['lspecies'][5] ???
        tr_stars = traced_stars_indices.sort()
        mask = np.in1d(dd[('stars', 'particle_index')], tr_stars)
        assert len(dd[('stars', 'particle_index')][mask]) == len(tr_stars)
        assert (dd[('stars', 'particle_index')][mask[0]]-ds.parameters['lspecies'][5]) == tr_stars[0]

        s_x = (dd[('stars', 'particle_position_x')][mask]-plot_center_pos[0]).in_units('kpc')
        s_y = (dd[('stars', 'particle_position_y')][mask]-plot_center_pos[1]).in_units('kpc')
        s_z = (dd[('stars', 'particle_position_z')][mask]-plot_center_pos[2]).in_units('kpc')
        del(dd)
        
    if ( ([] != traced_dm_indices) or ([] != traced_gas_indices) ):
        raise NotImplementedError
    
    particles_param = (star_particles_x, star_particles_y, star_particles_z, star_particles_mass, plt.cm.YlOrRd_r)
    #print 'BLAAAAAAAAAAAAAAAAAA:','\n',title_star,particles, plt_range, sgals_param, Rvir, show, vmin, vmax
    plot_projection(title_star, particles_param, plt_range, sgals_param, r_vir=Rvir, show=show, vmin=vmin, vmax=vmax)
    particles_param = (dm_particles_x, dm_particles_y, dm_particles_z, dm_particles_mass, plt.get_cmap('winter')) #winter_r plt.get_cmap('gnuplot_r') plt.cm.GnBu
    plot_projection(title_dm, particles_param, plt_range, sgals_param, r_vir=Rvir, show=show, vmin=vmin, vmax=vmax)
    particles_param = (gas_particles_x, gas_particles_y, gas_particles_z, gas_particles_mass, plt.get_cmap('ocean_r')) 
    plot_projection(title_gas, particles_param, plt_range, sgals_param, r_vir=Rvir, show=show) #summer_r

    # ALL
#    particles_param = (np.concatenate((star_particles_x, dm_particles_x, gas_particles_x)), \
#                       np.concatenate((star_particles_y, dm_particles_y, gas_particles_y)), \
#                       np.concatenate((star_particles_z, dm_particles_z, gas_particles_z)), \
#                       np.concatenate((star_particles_mass, dm_particles_mass, gas_particles_mass)), \
#                       plt.cm.GnBu) #
#    plot_projection(title_total, particles_param, plt_range, sgals_param, r_vir=Rvir, show=show, vmin=vmin, vmax=vmax)
    # TODO: uncomment
    

#Todo add hold?? what was that for?
def plot_projection(title, (p_x, p_y, p_z, p_m, p_cmap), plt_range, (g_x, g_y, g_z, g_r, g_m, g_id, ),\
                    gal_cmap='summer', gal_coloring='random', r_vir=0, colored_par_pos = ([],[],[]), show=True, vmin=3, vmax=9):
    if  (([],[],[]) != colored_par_pos):
        s_x, s_y, s_z = colored_par_pos

    
    plt.ioff()
    print "alive @ plot_projection" 
    
    f=plt.figure(1, figsize=(40,40)); grid_size = 1000 #should be 20,20 500
    plt.subplots_adjust(top=0.90, wspace=0, hspace=0)
    matplotlib.rcParams.update({'font.size': 20})
    
    # --- Set coloring scales: ----------------------------------->
    # cmaps ='hot' 'jet' 'terrain' 'Set1' 'Haze' 'Mac Style' 'summer' 'spring' 'winter'    
    if 'random' == gal_coloring: 
        g_colors = np.random.rand(g_id)
    elif 1<len(gal_mass): #logM coloring
        g_colors = np.log10(gal_mass/(max(gal_mass)-min(gal_mass)))
    else: # one gal so no galaxy
        #FIXME - bad practice
        g_colors=[1,]    
    scalarMap = cm.ScalarMappable(cmap=plt.get_cmap(gal_cmap, lut=8) )
    colorVal = scalarMap.to_rgba(g_colors)
    
   
    # Y - X :
    # =======
    plt.subplot(221)
    plt.gca().add_artist(plt.Circle((0,0), radius=r_vir, lw=5, color='grey', alpha=0.2, fill=True))
    print r_vir
    plt.gca().add_artist(plt.Circle((0,0), radius=0.1*r_vir, lw=5, color='grey', alpha=0.2, fill=True))
    scat = plt.hexbin(p_x,p_y, C=p_m, cmap=p_cmap, gridsize=grid_size,bins='log',alpha=1,reduce_C_function=np.sum,\
                      hold=True, vmin=vmin, vmax=vmax)
    # colored_par_pos part:
#     plt.hexbin(s_x, s_y, gridsize = 100, alpha=1, mincnt=1, reduce_C_function=np.sum, hold=True, color='blue', cmap=plt.get_cmap('Blues'), bins='log', vmin=10**-100) #cmap=plt.get_cmap('Blues') #cmap=plt.get_cmap('Blues')
    plt.axis([-plt_range, plt_range, -plt_range, plt_range])
    plt.grid(True)

    plt.subplot(221)
    # for code improvment see: http://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
    for i in range(g_id):
        plt.gca().add_artist(plt.Circle((g_x[i], g_y[i]), radius=g_r[i], lw=7, color=colorVal[i], alpha=0.9, fill=False))
#        plt.annotate('%02.2f kpc' % g_r[i], xy=(g_x[i], g_y[i]+g_r[i]), fontsize=15 )
    plt.xlabel('x [kpc]')
    plt.ylabel('y [kpc]')

    
    # X - Z :
    # =======
    plt.subplot(224)
    plt.gca().add_artist(plt.Circle((0,0), radius=r_vir, lw=5, color='grey', alpha=0.2, fill=True))
    plt.gca().add_artist(plt.Circle((0,0), radius=0.1*r_vir, lw=5, color='grey', alpha=0.2, fill=True))
    scat = plt.hexbin(p_z,p_x, C=p_m, cmap=p_cmap, gridsize=grid_size,bins='log',alpha=1,reduce_C_function=np.sum,\
                      hold=True, vmin=vmin, vmax=vmax)
#     plt.hexbin(s_z, s_x, gridsize = 100, alpha=1, mincnt=1, reduce_C_function=np.sum, hold=True, color='blue', cmap=plt.get_cmap('Blues'), bins='log', vmin=10**-30)
    plt.axis([-plt_range, plt_range, -plt_range, plt_range])
    plt.grid(True)

    plt.subplot(224)
    for i in range(g_id):
        plt.gca().add_artist(plt.Circle((g_z[i], g_x[i]), radius=g_r[i], lw=7, color=colorVal[i], alpha=0.9, fill=False))
#        plt.annotate('%02.2f kpc' % g_r[i], xy=(g_z[i], g_x[i]+g_r[i]), fontsize=15 )
    plt.xlabel('z [kpc]')
    plt.ylabel('x [kpc]')


    # Y - Z :
    # =======
    plt.subplot(222)
    plt.gca().add_artist(plt.Circle((0,0), radius=r_vir, lw=5, color='grey', alpha=0.2, fill=True))
    plt.gca().add_artist(plt.Circle((0,0), radius=0.1*r_vir, lw=5, color='grey', alpha=0.2, fill=True))
    scat = plt.hexbin(p_z,p_y, C=p_m, cmap=p_cmap, gridsize=grid_size,bins='log',alpha=1,reduce_C_function=np.sum,\
                      hold=True, vmin=vmin, vmax=vmax)
#     plt.hexbin(s_z, s_y, gridsize = 100, alpha=1, mincnt=1, reduce_C_function=np.sum, hold=True, color='blue', cmap=plt.get_cmap('Blues'), bins='log', vmin=10**-30) #cmap=plt.get_cmap('Blues')
    plt.axis([-plt_range, plt_range, -plt_range, plt_range])
    plt.grid(True)

    plt.subplot(222)
    for i in range(g_id):
        plt.gca().add_artist(plt.Circle((g_z[i], g_y[i]), radius=g_r[i], lw=7, color=colorVal[i], alpha=0.9, fill=False))
#        plt.annotate('%02.2f kpc' % g_r[i], xy=(g_z[i], g_y[i]+g_r[i]), fontsize=15 )


    cax = f.add_axes([0.13, 0.01, 0.7, 0.02])
    cbar = f.colorbar(scat, cax, orientation='horizontal', ticks=np.linspace(vmin, vmax, num=2*(vmax-vmin)+1, endpoint=True))
    cbar.set_label('$M\odot$', fontsize=30)
    aa = plt.suptitle(title, fontsize=30)
    #.set_title('$M_{baryon}\ vs\ M_{dm}\  $'+ data_filter_title, fontsize=t_fontsize, y=1.05)
    f.savefig('proj 40c.' + str(plt_range) + 'kpc.'+ title + ".png", format='png', \
              bbox_extra_artists=(cax,aa), bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()

    # Old optional code:
    # ==================
    # Scatter usage (do problems with plotting the right radius of the circles, needs a transformation func):
    # g_area =  map(lambda g: ( np.pi *10* (ds.quan(g.r,'kpc'))**2 ), gals) # 10 for fisoze(20,20)

    # plt.scatter(g_z, g_y, s=g_area, c=g_colors, alpha=0.35) # full circles
    # plt.scatter(g_x, g_y, s=g_area, c=g_colors, marker='o',facecolors='none', alpha=0.5, hold=True, linewidths =5)

    # --- coloring --------------------------------------------------------------------
    # cmap=plt.cm.YlOrRd_r = heat | RdYlBu_r = nice blue to red
    # cmap=plt.cm.jet
    

    
    