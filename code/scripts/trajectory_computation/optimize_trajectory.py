"""This code computes the optimal needle trajectory for 30 degrees needle inclination"""
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from os import path, pardir
import matplotlib as plt
from mpl_toolkits.mplot3d import proj3d
from csv import writer
import time

# Paths
skin_paths = "Replace with path to skin surface points file (csv file with all the surface points)"
skeleton_paths = "Replace with path to skeleton points file (csv file with all the surface points)"
opt_pose_path = "Write the path where results will be written"

# Load phantom surface points file 
surface_points = np.genfromtxt(skin_paths,dtype=float ,delimiter = ',')

print(np.min(surface_points[:,0]))

# Load skeleton pc
skeleton = np.genfromtxt(skeleton_paths,dtype=float ,delimiter = ',')
skeleton_om = np.hstack((skeleton, np.ones((len(skeleton),1))))
skeleton_om_n = np.zeros((len(skeleton_om),4))

# *****************************************************************************************
# ******************************* USER FUNCTIONS ******************************************
# *****************************************************************************************

# Define a function to get z_inf from x and y of surface
def get_y_surface(x_surf,z_surf):
    abs_tolerance = 0.001
    a = np.where(abs(x_surf-surface_points[:,0]) < abs_tolerance)
    b = np.where(abs(z_surf-surface_points[:,2]) < abs_tolerance)
    c = np.intersect1d(a[0],b[0])

    sum = 0
    for i in range(len(c)):
        sum = sum + surface_points[c[i],1]
    
    if(len(c) == 0):
        #print(x_surf,z_surf)
        return 1000

    else:
        return sum / (len(c))

# *****************************************************************************************
# **************************** GEOMETRICAL FEATURES ***************************************
# *****************************************************************************************

# Calyx insertion point
cal_point = np.array((-0.0743,-0.0340,0.0461,1.0))
best_dir = np.array((0.6530,0.3177,0.6875))

# Gemoetrical features of probe. 
# "d" points are the insertion points in the needle guide.
# "h" point is the exit point in the needle guide

d30_p = np.array((-0.08592, 0, -0.06381, 1.0))          
d37_p = np.array((-0.0896, 0, -0.0589))
d45_p = np.array((-0.0927, 0, -0.05364))
h_p = np.array((-0.06300, 0, -0.02412, 1.0))

def rotmatrix(v):
    return R.from_euler('ZYZ', v).as_matrix()

# *********************************** BOUNDS **********************************************
# Set bounds for optimization problem
x_min = np.min(surface_points[:,0]) + 0.002
y_min = np.min(surface_points[:,1]) + 0.002
z_min = np.min(surface_points[:,2]) + 0.002
x_max = np.max(surface_points[:,0]) - 0.002
y_max = np.max(surface_points[:,1]) - 0.002
z_max = np.max(surface_points[:,2]) - 0.002

b_lb = np.array((x_min, y_min, z_min, 0.0, 0.0, 0.0))
b_ub = np.array((z_max, y_max, z_max, 2*pi, 2*pi, 2*pi))
bounds = Bounds(lb=b_lb, ub=b_ub)

# Initial Pose for the alghorithm
x0 = np.array((cal_point[0],get_y_surface(cal_point[0],cal_point[2]),cal_point[2],pi/2,0,0))

# Compute poses with respect to mannequin frame
def get_h_m(x):
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]

    h_m = np.dot(T_p_m,h_p)
    return h_m

def get_incl_30(x):
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]
    d30_m = np.dot(T_p_m,d30_p)
    h_m = np.dot(T_p_m,h_p)
    incl_m = (h_m[0:3]-d30_m[0:3])/np.linalg.norm(h_m[0:3]-d30_m[0:3])

    return incl_m

''' ************************************************************************************************************
************************************************ CONSTRAINTS ***************************************************
************************************************************************************************************ '''

# *********************************************** TANGENCY ******************************************************
# Probe has to be tangent to skin surface. This condition can be obtained saying that the distance between the 
# probe arc center and the skin needs to be equal to probe radius. Lower bound is set negative to be sure that the
# probe is pressed on the skin. This constraint is equal for all the inclinations.

def tangency(x):
    arc_center = np.array((0.0,0.0,-0.0545,1.0))      # Polar coordinates of the center of probe's arc
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]
    arc_center_m = np.dot(T_p_m, arc_center)
    y_skin = get_y_surface(x[0],x[2])
    skin_point = np.array((x[0],y_skin,x[2]))

    return np.linalg.norm(arc_center_m[0:3] - skin_point) - 0.0545

tangency_constraint = NonlinearConstraint(fun=tangency, lb=-0.005, ub=0.0)


# **************************************** CALYX INTERSECTION *********************************************

def distance(x):

    # Cal respect to mannequin -> Cal respect to probe
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]
    cal_p = np.dot(np.linalg.inv(T_p_m),cal_point)

    # Cal respect to probe -> Stone respect to needle
    T_n_p = np.eye(4)
    T_n_p[0:3,0:3] = R.from_euler('y', pi/6).as_matrix()
    T_n_p[0:3,3] = d30_p[0:3]
    T_p_n = np.linalg.inv(T_n_p)

    cal_n = np.dot(T_p_n,cal_p)

    return np.sqrt(cal_n[0]**2 + cal_n[1]**2)

distance_constraint = NonlinearConstraint(fun=distance, lb=0.0, ub=0.001)
   
# **************************************** AVOID ORGANS COLLISION *********************************************
# Given the skeleton point cloud, I need to be sure that the minimum distance between the needle trajectory and
# theese points is greater than the needle radius (0.65 mm). We will set this bound distance grater than 0.65 to
# be more confidential. This constraint is different for the 3 inclinations.

def collision_30(x):
    # Skeleton respect to mannequin -> Skeleton respect to probe
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]
    T_m_p = np.linalg.inv(T_p_m)

    # Skeleton respect to probe -> Skeleton respect to needle
    T_n_p = np.eye(4)
    T_n_p[0:3,0:3] = R.from_euler('y', pi/6).as_matrix()
    T_n_p[0:3,3] = d30_p[0:3]
    T_p_n = np.linalg.inv(T_n_p)

    cal_p = np.dot(np.linalg.inv(T_p_m),cal_point)
    cal_n = np.dot(T_p_n,cal_p)
    min_dist = 1.0
    
    for point in range(len(skeleton_om)):
        skeleton_om_n[point,:] = np.dot(T_p_n, np.dot(T_m_p,skeleton_om[point,:]))

    c = np.intersect1d(np.where(np.abs(skeleton_om_n[:,0]) < 0.006),np.where(np.abs(skeleton_om_n[:,1]) < 0.006)) 

    for i in range(len(c)):
        if skeleton_om_n[c[i],2] <= cal_n[2]:
            dist = np.sqrt(skeleton_om_n[c[i],0]**2 + skeleton_om_n[c[i],1]**2)
            if dist < min_dist: 
                min_dist = dist

    return min_dist
   

collision_30_constraint = NonlinearConstraint(fun=collision_30, lb=0.005, ub=2.0)

# *****************************************************************************************
# ********************************** MINIMIZE *********************************************
# *****************************************************************************************

def function_to_minimize(x):
    incl = get_incl_30(x)
    cos_diff = np.dot(incl,best_dir)
    # alpha = np.rad2deg(np.arccos(cos_diff))
    print(incl)
    
    return (1-np.abs(cos_diff))

# Funzione di callback per fermare la minimizzazione
def my_callback(xk):
    if function_to_minimize(xk) < 0.01:
        return True  


# Optimization for 30° insertion
opzioni = {"maxiter" : 1000}
x30 = minimize(fun=function_to_minimize, x0=x0, bounds=bounds, constraints=[tangency_constraint, distance_constraint, collision_30_constraint], callback=my_callback, options=opzioni)
print("Solution:\n",x30)
print("Tangency Value = ", tangency(x30.x))
print("Distance Value = ", distance(x30.x))
print("Collision Value = ", collision_30(x30.x))
print("Direction = ", )

# ****************************************************************************************
# Write Pose
pose_30 = np.array((x30.x[0], x30.x[1], x30.x[2], x30.x[3], x30.x[4], x30.x[5])) 

# Write to file
with open(opt_pose_path, 'w', newline='') as my_csvfile:
    spamwriter = writer(my_csvfile,delimiter=',')
    spamwriter.writerow([pose_30[0]] + [pose_30[1]] + [pose_30[2]] + [pose_30[3]] + [pose_30[4]] + [pose_30[5]])