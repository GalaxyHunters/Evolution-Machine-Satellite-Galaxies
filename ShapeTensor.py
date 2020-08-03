import yt
from yt .mods import *
from yt import YTArray
import pandas as pd
import numpy as np
import struct
from scipy import interpolate
from scipy import optimize
import numpy.linalg as la
import bisect
from yt.units import kpc
from yt.units import Msun
from yt.units import Myr
import sys

def compute_shape(points_ref,m_stars,do_shell=False): 
    """finds shape tensor and returns eigen values and vectors of that tensor. 
    points_ref is an array with all x y z coordinantes (three coloumns of x first to z). 
    m_stars is a list of masses of the particles."""
    I = np.zeros((3,3))
    for k in [0,1,2]:
        for l in [0,1,2]:
            I[l][k]=sum(points_ref[:,k]*points_ref[:,l]*m_stars)
    eig_vals=la.eigvals(I)/(sum(m_stars))
    sorting = np.argsort(eig_vals)
    factor = np.sqrt(5.)
    if(do_shell): factor = sqrt(3.)
    cc,bb,aa = np.sqrt(eig_vals)[sorting]*factor
    eig_vecs = la.eig(I)[1]
    c_vec,b_vec,a_vec = eig_vecs[sorting]
    return cc,bb,aa,c_vec,b_vec,a_vec

def compute_shape_iter(points_ref,m_stars,r_eff,px_min=50.0,tol=0.05,max_iter=15,do_shell=False,min_shell=0.97,verbose=True,rescale=True):
    """finds best fitting shape tensor and returns eigen values and vectors of that tensor.
     points_ref is an array with all x y z coordinantes (three coloumns of x first to z).
     m_stars is a list of masses of the particles.
     tol--> decides how small of a jump in p and q is convergence
     max_iter is the number of times the function will try to fit a shape (defult 100)
     do_shell --> if you want to look at shell (true if so)
     min_shell --> if do_shell==true the func will look at a shell and min shell is the small ellipsoid in the inner part of the shell
     verbose--> more information 
     r_eff is the effective radius (defult 3.2)
     rescale --> decides if the eigen basis rescales in every iteration"""
#    box_x,box_y,box_z = array([1.,0.,0.]),array([0.,1.,0.]),array([0.,0.,1.])
    rot_basis = np.identity(3) #rotation basis in begining, changes soon
    #making spherical mask by taking only points with rad < r_eff
    points_ref_tmp=np.transpose(points_ref)
    points_ref_radii=np.sqrt(points_ref_tmp[0]**2+points_ref_tmp[1]**2+points_ref_tmp[2]**2)
    mask_r = [points_ref_radii <= r_eff]
    m_stars = m_stars[mask_r]
    points_ref = points_ref[mask_r]
    particles_start=len(points_ref)
    print(particles_start)
    
    c,b,a,c_vec,b_vec,a_vec=compute_shape(points_ref,m_stars,do_shell=do_shell) #finds eigen vectors and values of shape tensor
    scale_eigen = r_eff / a #the rescaleing factor
    if(rescale==False): scale_eigen = 1. # decides if the eigen values rescales 
    a *= scale_eigen # rescaling
    b *= scale_eigen
    c *= scale_eigen
    new_basis = np.transpose([a_vec,b_vec,c_vec]) # new basis in old basis coordinantes
    points_ref = np.dot(points_ref,new_basis) # new coordinates
    p_old = points_ref    
    rot_basis = np.dot(rot_basis,new_basis) # new rotation matrix
#    box_x = dot(box_x,new_basis)
#    box_y = dot(box_y,new_basis)
#    box_z = dot(box_z,new_basis)
    ratios_old = [b/a,c/b] # p and q (from articles) of spheroid
    iter=0 # to control iterations
    iterations=0
    err_ratios = 1. # to measure convergents
    a_dummy = []; b_dummy = []; c_dummy = [] #list of eigen values in different iterations
    a_vec_dummy = []; b_vec_dummy = []; c_vec_dummy = [] # list of eigen vectors in different iterations
    errors=[]#list of errors
    while((err_ratios>=tol)&(iter<100)):
        # rotate the points in the eigenvector base
        if(do_shell==False): #if false --> takes all points in the ellipsoid (by current a,b,and c)
            mask = (points_ref[:,0]/a)**2.+(points_ref[:,1]/b)**2.+(points_ref[:,2]/c)**2.<=1
        else: #if true --> takes points in shell from ellipsoid edge to a shell in middle determend by min_shell
            mask = ((points_ref[:,0]/a)**2.+(points_ref[:,1]/b)**2.+(points_ref[:,2]/c)**2.<=1)&((points_ref[:,0]/a)**2.+(points_ref[:,1]/b)**2.+(points_ref[:,2]/c)**2.>=min_shell)
        c_new,b_new,a_new,c_vec_new,b_vec_new,a_vec_new=compute_shape(points_ref[mask],m_stars[mask],do_shell=do_shell) #computes new eigen basis
        ratios_new = [b_new/a_new,c_new/b_new] #new p and q
        err_ratios = max(abs(ratios_old[0]-ratios_new[0])/(ratios_old[0]+ratios_new[0]),abs(ratios_old[1]-ratios_new[1])/(ratios_old[1]+ratios_new[1])) # differance in p and q. if small enough-->convergence
        #
        errors.append(err_ratios)
        if(verbose==True): 
            print("scale, err, radii = ",scale_eigen.round(2),err_ratios.round(2), len(points_ref[mask]),b,c)
        ratios_old = ratios_new #changing value for next loop
        if(do_shell==False): # dont know why
            scale_eigen = r_eff / a_new
        else:
            scale_eigen = r_eff / a_new
        #
        if(rescale==False): 
            scale_eigen = 1.        
        a = a_new * scale_eigen 
        b = b_new * scale_eigen
        c = c_new * scale_eigen
        a_vec = a_vec_new
        b_vec = b_vec_new
        c_vec = c_vec_new
        new_basis = np.transpose([a_vec,b_vec,c_vec])
        points_ref = np.dot(points_ref,new_basis)
        #px_err_ratio that we discussed with tomer
        special_mask = (points_ref[:,0]/a)**2.+(points_ref[:,1]/b)**2.+(points_ref[:,2]/c)**2.<=1
        p_new = points_ref[special_mask]
        if iter==0:
            p_old = p_old[mask]
        inter_sect = points_ref[np.logical_and(special_mask, mask)]
        if float(len(inter_sect))!=0:
            px_err_ratio = float(abs(len(p_old)-len(inter_sect))+abs(len(p_new)-len(inter_sect)))/float(len(inter_sect))
        if float(len(inter_sect))==0 & iter==0:
            px_err_ratio = 0.
        old_mask = special_mask        
        p_old=p_new
        print('px_err_ratio: ', px_err_ratio)     

        a_vec = np.dot(a_vec,new_basis)
        b_vec = np.dot(b_vec,new_basis)
        c_vec = np.dot(c_vec,new_basis)
#        box_x = dot(box_x,new_basis)
#        box_y = dot(box_y,new_basis)
#        box_z = dot(box_z,new_basis)
        #
        rot_basis = np.dot(rot_basis,new_basis)
        if la.cond(rot_basis) < 1/sys.float_info.epsilon:
            i = la.inv(rot_basis)
        else:
            px = [[particles_start],[np.nan]]
            err_ratios = [np.nan]
            err_msg = 'LinAlgErr: Singular Matrix'
            print err_msg
            var = [np.nan,np.nan,np.zeros((3)),np.zeros((3)),np.zeros((3)),err_ratios,px,np.nan,np.nan,err_msg]
            temp = [np.zeros((3)),np.zeros((3)),np.zeros((3))]
            return var,temp
            continue
        a_vec = np.dot(a_vec,la.inv(rot_basis))
        b_vec = np.dot(b_vec,la.inv(rot_basis))
        c_vec = np.dot(c_vec,la.inv(rot_basis))
#        box_basis = transpose([box_x,box_y,box_z])
#        a_vec = dot(a_vec,box_basis)
#        b_vec = dot(b_vec,box_basis)
#        c_vec = dot(c_vec,box_basis)
        iter+=1
        iterations+=1
        p = b/a
        q = c/b
        if (len(points_ref[mask])<=px_min):
            print("Error! px_amount<px_min!")
            err_msg="px_amount<px_min"
            #a,b,c,err_ratios=[np.nan]*4
            #a_vec,b_vec,c_vec=[((np.nan,np.nan,np.nan))]*3
            ##return last variables of function
            px = [[particles_start],[len(points_ref[mask])]]
            err_ratios_change=[[min(errors)],[err_ratios]]
            var = [p,q,a_vec,b_vec,c_vec,err_ratios_change,px,px_err_ratio,iterations,err_msg]
            temp = [a_vec,b_vec,c_vec]
            return var,temp

            continue
    if(iter>=max_iter):
        print("Error! iter>max_iter!")
        err_msg="iter>max_iter"
        errors = np.array(errors)
        err_ratios_change=[[min(errors)],[err_ratios]]
        px = [[particles_start],[len(points_ref[mask])]]
        var= [p,q,a_vec,b_vec,c_vec,err_ratios_change,px,px_err_ratio,iterations,err_msg]
        temp = [a_vec,b_vec,c_vec]
        return var,temp
    else:
        err_msg="All good"
        print("radii: a=%0.02f, b=%0.02f, c=%0.02f " % (a,b,c))
        print("vectors: a, b, c = ", a_vec,b_vec,c_vec)
        print("err_ratios = ", err_ratios)
        print("err_msg = ", err_msg)
        px = [particles_start],[len(points_ref[mask])]
        err_ratios = [err_ratios]
        #var = [p,q,a_vec,b_vec,c_vec,err_ratios,px,px_err_ratio,iterations,err_msg]
        var = [a_vec,b_vec,c_vec,err_ratios,px,err_msg]
        temp = [p,q,px_err_ratio,iterations]
        return var,temp
    
def imp_data(sgal_cat_s,central=False):
    if central==False:
        sp,center=get_sp_sgal(sgal_cat_s)
    else:
        sp,vel,center=get_sp_cgal(sgal_cat_s)
    print("Importing star data")
    starpx=sp[('stars', 'particle_position_x')].in_units('kpc')-center[0]
    starpy=sp[('stars', 'particle_position_y')].in_units('kpc')-center[1]
    starpz=sp[('stars', 'particle_position_z')].in_units('kpc')-center[2]
    #defining the objects that go in the compute_shape_iter function
    mat_star=np.transpose((starpx.v,starpy.v,starpz.v))
    starm=sp[('stars', 'particle_mass')].in_units('Msun').v

    print("Importing dark matter data")
    darkpx=sp[('darkmatter', 'particle_position_x')].in_units('kpc')-center[0]
    darkpy=sp[('darkmatter', 'particle_position_y')].in_units('kpc')-center[1]
    darkpz=sp[('darkmatter', 'particle_position_z')].in_units('kpc')-center[2]
    mat_dark=np.transpose((darkpx.v,darkpy.v,darkpz.v))
    darkm=sp[('darkmatter', 'particle_mass')].in_units('Msun').v

    print("Importing gas data")
    gaspx=sp[('index', 'x')].in_units('kpc')-center[0]
    gaspy=sp[('index', 'y')].in_units('kpc')-center[1]
    gaspz=sp[('index', 'z')].in_units('kpc')-center[2]
    mat_gas=np.transpose((gaspx.v,gaspy.v,gaspz.v))
    gasd=sp[('gas', 'density')].in_units('Msun/kpc**3')
    vol=sp[('index', 'cell_volume')].in_units('kpc**3')
    gasm=np.array(gasd*vol)
    
    '''print('Importing Youngstar data')   
    stars_age = [sp[('stars','particle_age')].in_units('Myr').v <= 100]
    youngstarm = starm[stars_age]    
    youngstarpx=starpx[stars_age]
    youngstarpy=starpy[stars_age]
    youngstarpz=starpz[stars_age]
    mat_youngstar=np.transpose((youngstarpx.v,youngstarpy.v,youngstarpz.v))'''
    
    print("Importing Coldgas data")
    cold_mask = [sp[('gas', 'temperature')].in_units('K').v <= 1.5*10**4]
    coldgaspx = gaspx[cold_mask]
    coldgaspy = gaspy[cold_mask]
    coldgaspz = gaspz[cold_mask]
    mat_coldgas=np.transpose((coldgaspx.v,coldgaspy.v,coldgaspz.v))
    coldgasm = gasm[cold_mask]
    
    print("Importing total matter data")
    allpx=np.concatenate((starpx, darkpx, gaspx))
    allpy=np.concatenate((starpy, darkpy, gaspy))
    allpz=np.concatenate((starpz, darkpz, gaspz))
    mat_tot=np.transpose((allpx.v,allpy.v,allpz.v))
    totm=np.concatenate((starm, darkm, gasm))

    mat_arr = [mat_star,mat_dark,mat_gas,mat_coldgas,mat_tot]
    mass_arr = [starm,darkm,gasm,coldgasm,totm]
        
    return mat_arr,mass_arr

def shapetensor(mat_arr,mass_arr,r):
    print("Stars") 
    star_vars,temp_0 = compute_shape_iter(mat_arr[0],mass_arr[0],r)
    print("Darkmatter")
    dm_vars,temp_1 = compute_shape_iter(mat_arr[1],mass_arr[1],r)
    print("Gas")
    gas_vars,temp_2 = compute_shape_iter(mat_arr[2],mass_arr[2],r)
    '''if len(mass_arr[3])>=1:
        print("Youngstars")
        youngstar_vars = compute_shape_iter(mat_arr[3],mass_arr[3],r)
    else:
        px = [[len(mass_arr[3])],[np.nan]]
        err_ratios = [np.nan]
        var = [np.nan,np.nan,np.zeros((3)),np.zeros((3)),np.zeros((3)),err_ratios,px,np.nan,np.nan,'Not Enough Particles']
        youngstar_vars = var'''
    if len(mass_arr[3])>=1:
        print("Coldgas")
        coldgas_vars,temp_3 = compute_shape_iter(mat_arr[3],mass_arr[3],r)    
    else:
        px = [[len(mass_arr[3])],[np.nan]]
        err_ratios = [np.nan]
        var = [np.nan,np.nan,np.zeros((3)),np.zeros((3)),np.zeros((3)),err_ratios,px,np.nan,np.nan,'Not Enough Particles']
        coldgas_vars = var
    print('Total')
    tot_vars,temp_4 = compute_shape_iter(mat_arr[4],mass_arr[4],r)    

    return star_vars,dm_vars,gas_vars,coldgas_vars,tot_vars,temp_0,temp_1,temp_2,temp_3,temp_4

def get_sp_sgal(sgal_cat_s):    
    cen_col=['x[kpc]', 'y[kpc]', 'z[kpc]']
    radius=sgal_cat_s['Rsat[kpc]']
    center2=sgal_cat_s[cen_col].values
    center3=center2.astype('Float64')
    center = ds.arr(center3, 'kpc')
    sp = ds.sphere(center=center,radius =(9*radius,'kpc'))
    return sp,center

def get_sp_dm(sgal_cat_s, rad_flag = 0):    
    cen_col=['x[kpc]', 'y[kpc]', 'z[kpc]']
    radius=sgal_cat_s['Rsat[kpc]']
    center2=sgal_cat_s[cen_col].values
    center3=center2.astype('Float64')
    center = ds.arr(center3, 'kpc')
    if rad_flag !=0:
        sp = ds.sphere(center=center,radius =(float(rad_flag), 'kpc'))
    else:
        sp = ds.sphere(center=center,radius =(15*radius,'kpc'))
    return sp

def get_sp_dm_outdated(sgal_cat_s, rad_flag = 0):    
    cen_col=['zana_x','zana_y','zana_z']
    radius=float(sgal_cat_s['Reff(Rsb)'])*2
    center2=sgal_cat_s[cen_col].values
    center3=center2.astype('Float64')
    center = ds.arr(center3, 'code_length')
    if rad_flag !=0:
        sp = ds.sphere(center=center,radius =(float(rad_flag), 'kpc'))
    else:
        sp = ds.sphere(center=center,radius =(15*radius,'kpc'))
    return sp

def get_sp_am(sgal_cat_s):  
    radius=sgal_cat_s['Rsat[kpc]']
    cen_col=['x[kpc]', 'y[kpc]', 'z[kpc]']
    center2=sgal_cat_s[cen_col].values
    center3=center2.astype('Float64')
    center = ds.arr(center3, 'kpc')
    sp = ds.sphere(center=center,radius=(9*radius,'kpc'))
    
    v_col=['Vx[km/s]','Vy[km/s]','Vz[km/s]']
    v2=sgal_cat_s[v_col].values
    v=v2.astype('Float64')
    vel = ds.arr(v, 'km/s')
    vel=vel.in_units('kpc/Myr')
    
    return sp,vel,center
   
def get_sp_cgal(cen_gal_cat_s, rad_flag = 0):
    cen_col=['center[0](code)', 'center[1](code)', 'center[2](code)']
    rad=cen_gal_cat_s['r_vir[kpc]']
    center2=cen_gal_cat_s[cen_col].values
    center= ds.arr(center2, 'code_length').in_units('kpc')
    if rad_flag !=0:
        sp = ds.sphere(center=center,radius =(float(rad_flag), 'kpc'))
    else:
        sp = ds.sphere(center=center,radius =(15*rad,'kpc'))
    
    #v_col=['vcenter[0](km/s)', 'vcenter[1](km/s)', 'vcenter[2](km/s)']
    #v2 = cen_gal_cat_s[v_col].values
    #v=v2.astype('Float64')
    #vel = ds.arr(v, 'km/s')
    #vel=vel.in_units('kpc/Myr')
    
    return sp

def calc_am_all(mass_arr,v_arr,r_arr,rad):
    print('Stars')
    stars_am=calc_am(mass_arr[0],v_arr[0],r_arr[0],rad)
    print('darkmatter')
    dm_am=calc_am(mass_arr[1],v_arr[1],r_arr[1],rad)
    print('gasses')
    gas_am=calc_am(mass_arr[2],v_arr[2],r_arr[2],rad)
    #print('tot_m')
    #mass_arr=np.concatenate([mass_arr[0],mass_arr[1],mass_arr[2]])
    #v_arr=np.concatenate([v_arr[0],v_arr[1],v_arr[2]])
    #r_arr=np.concatenate([r_arr[0],r_arr[1],r_arr[2]])
    #totm_am=calc_am(mass_arr,v_arr,r_arr,rad)
    
    return stars_am,dm_am,gas_am#,totm_am

#func that numpy doesn't have for some reason
#a is matrix, b is list
hocuspocus = lambda a,b: [[r*q for r in p] for p, q in zip(a,b)]
   
def calc_am(mass,v,r,rad,mass_norm=False):
    #getting mask of particles from total particles
    #print('masking')
    mass_array=np.transpose(np.concatenate([[mass,mass,mass]]))
    r_temp = np.transpose(r)
    r_radii = np.sqrt(r_temp[0]**2+r_temp[1]**2+r_temp[2]**2)
    mask_r = [r_radii <= rad]
    mass_array = mass_array[mask_r]
    r = r[mask_r]
    v = v[mask_r]

    #print('Calculating AM', '\n')
    ang_mntm=np.transpose(np.cross(r,(v*mass_array)))
    if mass_norm==False:
        mass_norm = np.sum(mass_array)/3.0
    else:
        mass_norm = 1.0
    x=np.sum(ang_mntm[0])/mass_norm
    y=np.sum(ang_mntm[1])/mass_norm
    z=np.sum(ang_mntm[2])/mass_norm
    am=np.array([x,y,z])
    return am
          
def get_particle_data(sgal_cat_s,central=False):
    if central==False:
        sp,vel,center=get_sp_am(sgal_cat_s)
    else:
        sp,vel,center=get_sp_cgal(sgal_cat_s)
        
    star_mass,star_v,star_r=particle_data('stars',sp,vel,center)
    dm_mass,dm_v,dm_r=particle_data('darkmatter',sp,vel,center)
    gas_mass,gas_v,gas_r=particle_data('gas',sp,vel,center)
    mass_arr=[star_mass,dm_mass,gas_mass]
    v_arr = [star_v,dm_v,gas_v]
    r_arr = [star_r,dm_r,gas_r]
    return mass_arr,v_arr,r_arr
    
def particle_data(particle_type,sp,vel,center):
    if str(particle_type) == 'stars' or str(particle_type) == 'darkmatter':
        #print("Importing " + str(particle_type) +" data")
        rx=sp[(str(particle_type), 'particle_position_x')].in_units('kpc')-center[0]
        ry=sp[(str(particle_type), 'particle_position_y')].in_units('kpc')-center[1]
        rz=sp[(str(particle_type), 'particle_position_z')].in_units('kpc')-center[2]
    
        vx=sp[(str(particle_type), 'particle_velocity_x')].in_units('kpc/Myr')-vel[0]
        vy=sp[(str(particle_type), 'particle_velocity_y')].in_units('kpc/Myr')-vel[1]
        vz=sp[(str(particle_type), 'particle_velocity_z')].in_units('kpc/Myr')-vel[2]
        mass=np.array(sp[(str(particle_type), 'particle_mass')].in_units('Msun'))
    elif str(particle_type)=='gas':
        #print('Importing gas data')
        rx=sp[('index', 'x')].in_units('kpc')-center[0]
        ry=sp[('index', 'y')].in_units('kpc')-center[1]
        rz=sp[('index', 'z')].in_units('kpc')-center[2]
    
        vx=sp[('gas', 'velocity_x')].in_units('kpc/Myr')-vel[0]
        vy=sp[('gas', 'velocity_y')].in_units('kpc/Myr')-vel[1]
        vz=sp[('gas', 'velocity_z')].in_units('kpc/Myr')-vel[2]
        gasd=sp[('gas', 'density')].in_units('Msun/kpc**3')
        vol=sp[('index', 'cell_volume')].in_units('kpc**3')
        mass=np.array(gasd*vol)
    
    else:
        raise ValueError('Unknown particle_type was entered %s' % str(particle_type))
               
    r=np.transpose(np.array([rx,ry,rz]))
    v=np.transpose(np.array([vx,vy,vz])) 
              
    return mass,v,r
    
def r_per__m_dark_matter(sp, coeff, m_star = None):
    if m_star is None:
        m_star = sp.quantities.total_quantity(('darkmatter', 'particle_mass')).in_units('Msun')
    ninety_m_star = coeff*m_star
    star_order = np.argsort(sp[('darkmatter', 'particle_position_spherical_radius')])
    stars_r   = sp[('darkmatter', 'particle_position_spherical_radius')][star_order].in_units('kpc')
    star_cum_mass = np.cumsum( sp[('darkmatter', 'particle_mass')][star_order].in_units('Msun')) 
    ninety_m_star_i = bisect.bisect(star_cum_mass, ninety_m_star)
    if ninety_m_star_i==0:
        stars_r=np.nan
        return stars_r
    else:
        return stars_r [ninety_m_star_i-1].v
    
def return_rad(gal):
    sp_def = get_sp_dm_outdated(gal)
    r_der = find_DM_r_by_r_change_search(sp_def)
    r_1min = find_DM_1min_r_by_der(sp_def)
    sp_min = get_sp_dm_outdated(gal, min([r_der,r_1min]))
    r_dm = r_per__m_dark_matter(sp_min, 0.9)
    print r_dm
    if np.isnan(r_dm)==True:
        r_dm_half = np.nan
    else:
        sp_r_dm = get_sp_dm_outdated(gal, r_dm)
        r_dm_half = r_per__m_dark_matter(sp_r_dm, 0.5)
    print('r_der = ', r_der)
    print('r_1min = ', r_1min)
    print('r_dm = ', r_dm)
    print('r_dm_half = ',r_dm_half)
    return r_der,r_1min,r_dm,r_dm_half
    
# =====================================================================================>
   #The 2 radii to calculate, all you need is here.
# =====================================================================================>

# Only thing to change: min_bin_dis = for DM you can take 100 pc? and find_DM_1min_r_by_der(per_thresh = should be same as rvir. its (dln\rho)/(dln r) == per_thresh boundry)
# SP should be a sphere with 1.5 - 2 of the expected Rvir...
# in case of emergency: sv_w, sv_p = 41, 4 in find_DM_1min_r_by_der which talk about smoothing window for savgol, and the polonomyal degree (4 , try to change the other, this works)

#find_DM_1min_r_by_der(sp, per_thresh=1.01, min_bin_dis=0.025, xk=None, yk=None):
#find_DM_r_by_r_change_search(sp, per_thresh=0.1, min_bin_dis=0.025, xk=None, yk=None):
# =====================================================================================>


def get_mass_binned_DM(sp, min_bin_dis = 0.025):
    sp_r = sp.radius.in_units('kpc').v
    stars_r = sp[('darkmatter', 'particle_position_spherical_radius')].in_units('kpc').v
    stars_mass = sp[('darkmatter', 'particle_mass')].in_units('Msun').v
#     #cumsum
#     stars_r_argsorted = np.argsort(stars_r)
#     stars_mass_sorted = stars_mass[stars_r_argsorted]
#     stars_r_sorted    = stars_r[stars_r_argsorted]
#     stars_mass_cumsum = np.cumsum(stars_mass_sorted)
    # Binning (min_bin_dis pc bin)
    hist_bins = int(sp_r/min_bin_dis)
    hist_range = (0, sp_r) #was 100
    star_mass_binned, star_r_bin_edges = np.histogram(stars_r, bins=hist_bins,normed=False, range=hist_range, weights=stars_mass)
    stars_mass_binned_cumsum = np.cumsum(star_mass_binned)
    star_r_binning = 0.5*(star_r_bin_edges[1:]+star_r_bin_edges[:-1])
    yk = stars_mass_binned_cumsum
    xk = star_r_binning

#   star_volume = (4*np.pi/3) * (star_r_bin_edges[1:]**3 - star_r_bin_edges[:-1]**3)
#   star_density = np.divide(star_mass_binned, star_volume)
    return (xk,yk)



# Mass radius 0:
def find_DM_r_by_r_change_search(sp, per_thresh=0.1, min_bin_dis=0.025 , xk=None, yk=None): #min_bin_dis=0.025
    R_MIN_RES = min_bin_dis*8 # 0.2 # kpc = 200 pc

    sp_r = sp.radius.in_units('kpc').v
    stars_r = sp[('darkmatter', 'particle_position_spherical_radius')].in_units('kpc').v
    stars_mass = sp[('darkmatter', 'particle_mass')].in_units('Msun').v
    stars_r_argsorted = np.argsort(stars_r)
    stars_mass_sorted = stars_mass[stars_r_argsorted]
    stars_r_sorted    = stars_r[stars_r_argsorted]
    stars_mass_cumsum = np.cumsum(stars_mass_sorted)

    r_cur = 1.5*R_MIN_RES
    DR_MIN_RES = R_MIN_RES/8
    MAX_ITER = 1000
    #find the total_mass point to r_searched
    r_width = max(0.1*r_cur, R_MIN_RES)
    #print r_cur-.5*r_width
    m_cur_1 = stars_mass_cumsum[np.searchsorted(stars_r_sorted, r_cur-.5*r_width)-1]
    m_cur_2 = stars_mass_cumsum[np.searchsorted(stars_r_sorted, r_cur+.5*r_width)-1]
    slope = ( np.log10(m_cur_2)-np.log10(m_cur_1) )/ (np.log10(r_cur+.5*r_width) - np.log10(r_cur-.5*r_width))
    #slope = ( np.log10(m_cur_2)-np.log10(m_cur_1) )
    #print 'slope', slope
    while (per_thresh < slope and r_cur < 0.5*sp_r):
        #print '1.1*r_cur', r_cur
        #print 'slope', slope
        r_width = max(0.1*r_cur, R_MIN_RES) #TODO #TODO
        r_cur_1 = max(0.00001,    r_cur-.5*r_width)
        r_cur_2 = min(max(sp_r-1,0), r_cur+.5*r_width)
        m_cur_1 = stars_mass_cumsum[np.searchsorted(stars_r_sorted,  r_cur_1)-1]
        m_cur_2 = stars_mass_cumsum[np.searchsorted(stars_r_sorted,  r_cur_2)-1]
        slope = ( np.log10(m_cur_2)-np.log10(m_cur_1) )/ (np.log10(r_cur_2) - np.log10(r_cur_1))
        r_cur = 1.1*r_cur

    if  sp_r <= r_cur:
        slope = 0
        per_thresh = 0

        print('BLAT', slope, r_cur, sp_r)  # TODO RASIE EXCEPTION OR SOMTHING

    dr = .5*r_cur
    #r_cur = r_cur-dr
    #dr = .5*r_cur
    i=0
    epsi = 0.01*per_thresh
    while (DR_MIN_RES < dr and i < MAX_ITER and r_cur < sp_r):
        if (slope+epsi < per_thresh):
            #print 'slope < per_thresh', r_cur, slope, per_thresh
            r_cur = r_cur-dr
        elif (slope-epsi > per_thresh):
            #print 'per_thresh <= slope', r_cur, slope, per_thresh
            r_cur= r_cur+dr
        else:
            # THATS FINE, CHECK IF IT DOES HAPPEN AT ALL
            #print 'bla bla bla' # TODO RASIE EXCEPTION OR SOMTHING
            dr = 0
        r_cur_2 = min(max(sp_r-1,0), r_cur+.5*r_width)
        r_width = max(0.1*r_cur, R_MIN_RES)
        r_cur_1 = max(0.00001,    r_cur-.5*r_width)
        
        m_cur_1 = stars_mass_cumsum[np.searchsorted(stars_r_sorted,  r_cur_1)-1]
        m_cur_2 = stars_mass_cumsum[np.searchsorted(stars_r_sorted,  r_cur_2)-1]
        
        slope = ( np.log10(m_cur_2)-np.log10(m_cur_1)) / (np.log10(r_cur_2) - np.log10(r_cur_1))
        #slope = ( np.log10(m_cur_2)-np.log10(m_cur_1) )
        #print 'slope', slope
        dr = 0.5*dr
        i += 1

    if (MAX_ITER == i):
        print('MAX_ITER == i', MAX_ITER, i)

    return r_cur



# Rho_1st_min 1:
from scipy.signal import savgol_filter

def find_DM_1min_r_by_der(sp, per_thresh=1.01, min_bin_dis=0.025, xk=None, yk=None):
    sv_w, sv_p = 41, 4 #31 4
    #print 'find_1min_r_by_der'

    if (xk is None or yk is None):
        xk, yk = get_mass_binned_DM(sp, min_bin_dis)

    sg_fit_d1 = savgol_filter(yk, window_length=sv_w, polyorder=sv_p, \
                            deriv=1, mode='nearest', delta=min_bin_dis) # *10.0**-3
    sg_fit_log_rho_d2 = savgol_filter(np.log10((sg_fit_d1)/xk**2), window_length=sv_w, polyorder=3, \
                            deriv=1, mode='nearest', delta=min_bin_dis) # *10.0**-3
    #counter = 1
    for i in range(len(sg_fit_log_rho_d2)-1):
        # if the secound derivitive is closed enough to the 0
        if (abs(sg_fit_log_rho_d2[i]) < (per_thresh-1)):
            #counter = counter + 1
            return .5*(xk[i]+xk[i+1])

        # if the sign is changeing
        if ((sg_fit_log_rho_d2[i]) * (sg_fit_log_rho_d2[i+1]) < 0):
            #counter = counter + 1
            return .5*(xk[i]+xk[i+1])

    return np.inf
#print L
#print center
    
    
    
    
def norm_csi(points_ref,m_stars,tol=0.01,max_iter=100,do_shell=False,min_shell=0.97,verbose=False,r_eff=3.2,rescale=True):
    temp=compute_shape_iter(points_ref,m_stars,tol,max_iter,do_shell,min_shell,verbose,r_eff,rescale)
    a=1
    b=temp[1]/temp[2]
    c=temp[0]/temp[2]
    c_vec_size=np.sqrt(temp[3][0]**2+temp[3][1]**2+temp[3][2]**2)
    b_vec_size=np.sqrt(temp[4][0]**2+temp[4][1]**2+temp[4][2]**2)
    a_vec_size=np.sqrt(temp[5][0]**2+temp[5][1]**2+temp[5][2]**2)
    c_vec=temp[3]/c_vec_size
    b_vec=temp[4]/b_vec_size
    a_vec=temp[5]/a_vec_size
    return c,b,a,c_vec,b_vec,a_vec,temp[6]