"""This code computes the optimal needle trajectory for 30 degrees needle inclination"""
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from csv import writer
import open3d as o3d

# Paths
skin_paths = "Replace with path to skin surface points file (csv file with all the surface points)"
skeleton_paths = "Replace with path to skeleton points file (csv file with all the skeleton points)"
opt_pose_path = "Write the path where results will be written"

skin_paths = "/home/ars_control/Desktop/robotic_assisted_pcnl/data/ct/3d_reconstruction/segmentations/skin.csv"
skeleton_paths = "/home/ars_control/Desktop/robotic_assisted_pcnl/data/ct/3d_reconstruction/segmentations/skeleton.csv"
opt_pose_path = "/home/ars_control/Desktop/robotic_assisted_pcnl/data/ct/3d_reconstruction/segmentations/traj.csv"

# Load phantom surface points file 
surface_points = np.genfromtxt(skin_paths,dtype=float ,delimiter = ',') / 1000

# Load skeleton pc
skeleton = np.genfromtxt(skeleton_paths,dtype=float ,delimiter = ',') / 1000
skeleton_om = np.hstack((skeleton, np.ones((len(skeleton),1))))
skeleton_om_n = np.zeros((len(skeleton_om),4))

# *****************************************************************************************
# **************************** GEOMETRICAL FEATURES ***************************************
# *****************************************************************************************

# Calyx insertion point
cal_point = np.array((-0.022,0.172,0.126,1.0))
best_dir = np.array((0.76367532, 0.31112698, 0.56568542))

# Gemoetrical features of probe and needle-guide
d30_p = np.array((-0.08592, 0, -0.06381, 1.0))         # insertion point in the needle guide.  
h_p = np.array((-0.06300, 0, -0.02412, 1.0))           # exit point in the needle guide

# Return rotmatrix starting from euler angles zyz
def rotmatrix(v):
    return R.from_euler('ZYZ', v).as_matrix()

# Compute initial guess
def compute_x0(cal,dir):

    """ Z axis of needle is set equal to the best direction, while the other 2 orthonormal 
    directions are chosen randomly. Starting from needle orientation and position, it is
    possible to retrieve probe pose. We choose a pose that is tangent to skin."""

    # Find 2 orthonromal versors to best direction
    v1 = np.random.randn(3)  
    v1 -= v1.dot(dir) * dir      
    v1 /= np.linalg.norm(v1)  
    v2 = np.cross(dir,v1)

    # Needle transformation
    T_needle = np.eye(4)
    T_needle_probe = np.eye(4)
    T_needle_probe[:3,:3] = rotmatrix([0,pi/6,0])
    T_needle_probe[:3,3] = d30_p[:3]

    # Impose z-axis direction
    T_needle[:3,0] = v1
    T_needle[:3,1] = v2
    T_needle[:3,2] = dir
    T_needle[:3,3] = cal[:3]

    # Translation along -z until probe is tangent the skin
    T_probe = np.dot(T_needle,np.linalg.inv(T_needle_probe))
    dist = np.min(np.linalg.norm(surface_points - T_needle[:3,3], axis=1))

    while dist > 0.002:
        T_needle[:3,3] = T_needle[:3,3] - 0.002*dir
        T_probe = np.dot(T_needle,np.linalg.inv(T_needle_probe))
        dist = np.min(np.linalg.norm(surface_points - T_probe[:3,3], axis=1))
        print(dist)

    T_probe = np.dot(T_needle,np.linalg.inv(T_needle_probe))

    angles = R.from_matrix(T_probe[:3,:3]).as_euler('ZYZ')
    x0 = np.array((T_probe[0,3],T_probe[1,3],T_probe[2,3],angles[0],angles[1],angles[2]))

    return x0


# Define a function to get y coordinate starting from the other 2
def get_y_surface(x_surf,z_surf):
    abs_tolerance = 0.001
    a = np.where(abs(x_surf-surface_points[:,0]) < abs_tolerance)
    b = np.where(abs(z_surf-surface_points[:,2]) < abs_tolerance)
    c = np.intersect1d(a[0],b[0])

    sum = 0
    for i in range(len(c)):
        sum = sum + surface_points[c[i],1]
    
    if(len(c) == 0):
        return 1000

    else:
        return sum / (len(c))

# Visualize mannequin surface, skeleton, probe and needle frames, calyx insertion point
def visualize_probe_pose(x_probe, cal):

    global skeleton, surface_points

    # Skin PC
    skin_pc = o3d.geometry.PointCloud()
    skin_pc.points = o3d.utility.Vector3dVector(surface_points)
    skin_pc.paint_uniform_color([0.2, 0.0, 0.0]) 

    # Skeleton PC
    skel_pc = o3d.geometry.PointCloud()
    skel_pc.points = o3d.utility.Vector3dVector(skeleton)
    skel_pc.paint_uniform_color([0.3, 0.3, 0.3]) 

    # Probe frame
    T_probe = np.eye(4)
    T_probe[:3,:3] = rotmatrix(x_probe[3:6])
    T_probe[:3,3] = x_probe[:3]
    probe_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    probe_frame.transform(T_probe)

    # Needle frame
    T_needle = np.eye(4)
    T_needle[:3,:3] = rotmatrix([0,pi/6,0])
    T_needle[:3,3] = d30_p[:3]
    T = np.dot(T_probe,T_needle)
    needle_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    needle_frame.transform(T)

    # Calyx insertion point
    calyx = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    calyx.paint_uniform_color([0.1, 0.7, 0.7])
    calyx.translate(cal[:3])
    
    # Labels
    o3d.visualization.draw_geometries([skel_pc,skin_pc,probe_frame,needle_frame,calyx])


def get_incl_30(x):
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]
    d30_m = np.dot(T_p_m,d30_p)
    h_m = np.dot(T_p_m,h_p)
    incl_m = (h_m[0:3]-d30_m[0:3])/np.linalg.norm(h_m[0:3]-d30_m[0:3])

    return incl_m    

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

x0 = compute_x0(cal_point,best_dir)


''' ************************************************************************************************************
************************************************ CONSTRAINTS ***************************************************
************************************************************************************************************ '''

# *********************************************** TANGENCY ******************************************************
# Probe has to be tangent to skin surface. This condition can be obtained saying that the distance between the 
# probe arc center and the skin needs to be equal to probe radius. Lower bound is set negative to ensure that the
# probe is pressed on the skin.

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
# Needle line must intersect the calyx insertion point. This constrint is formulated saying that distance 
# between needle and calyx point is minimal

def distance(x):

    # Cal respect to mannequin -> Cal respect to probe
    T_p_m = np.eye(4)
    T_p_m[0:3,0:3] = rotmatrix(x[3:6])
    T_p_m[0:3,3] = x[0:3]
    cal_p = np.dot(np.linalg.inv(T_p_m),cal_point)

    print("T_cal_man = ", cal_p)

    # Cal respect to probe -> Cal respect to needle
    T_n_p = np.eye(4)
    T_n_p[0:3,0:3] = R.from_euler('y', pi/6).as_matrix()
    T_n_p[0:3,3] = d30_p[0:3]
    T_p_n = np.linalg.inv(T_n_p)

    cal_n = np.dot(T_p_n,cal_p)
    print("T_cal_n = ", cal_n)

    return np.sqrt(cal_n[0]**2 + cal_n[1]**2)

distance_constraint = NonlinearConstraint(fun=distance, lb=0.0, ub=0.001)
   
# **************************************** AVOID ORGANS COLLISION *********************************************
# Given the skeleton point cloud, we need to be sure that the minimum distance between the needle trajectory and
# theee points is greater than the needle radius (0.65 mm). We will set this bound distance greater than 0.65 to
# be more confident.

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

''' ************************************************************************************************************
************************************************ OPTIMIZATION **************************************************
************************************************************************************************************ '''

def function_to_minimize(x):
    incl = get_incl_30(x)
    cos_diff = np.dot(incl,best_dir)
    
    print(1-np.abs(cos_diff))
    return (1-np.abs(cos_diff))

# Callback function to stop minimization
def my_callback(xk):
    if function_to_minimize(xk) < 0.01:
        return True  

# Init values
print("Tangency = ", tangency(x0))
print("Point Intersection = ", distance(x0))
print("Collision = ", collision_30(x0))
print("Minimization = ", function_to_minimize(x0))
visualize_probe_pose(x0, cal_point)

# Optimization for 30° insertion
options = {"maxiter" : 1000}
x30 = minimize(fun=function_to_minimize, x0=x0, method='SLSQP', bounds=bounds, constraints=[tangency_constraint, distance_constraint, collision_30_constraint], callback=my_callback, options=options)
print("Solution:\n",x30)
print("Tangency Value = ", tangency(x30.x))
print("Distance Value = ", distance(x30.x))
print("Collision Value = ", collision_30(x30.x))

# ****************************************************************************************
# Write Pose
pose_30 = np.array((x30.x[0], x30.x[1], x30.x[2], x30.x[3], x30.x[4], x30.x[5])) 

visualize_probe_pose(pose_30, cal_point)

# Write to file
with open(opt_pose_path, 'w', newline='') as my_csvfile:
    spamwriter = writer(my_csvfile,delimiter=',')
    spamwriter.writerow([pose_30[0]] + [pose_30[1]] + [pose_30[2]] + [pose_30[3]] + [pose_30[4]] + [pose_30[5]])