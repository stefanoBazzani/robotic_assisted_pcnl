import open3d as o3d
import numpy as np
import pandas as pd
from math import inf

def stl_to_csv(mesh_path, output_path, pc_number=1000000, ds_voxel_size = 1.0, lx=-inf, ux=inf, ly=-inf, uy=inf, lz=-inf, uz=inf):

    """This function converts a 3D mesh stored in an STL file into a downsampled point cloud and exports the point coordinates to a CSV file.
        Parameters:     
        - mesh_path (str): Path to the input STL mesh file.
        - output_path (str): Path where the output CSV file will be saved.
        - pc_number (int, default=1,000,000): Number of points to uniformly sample from the mesh surface.
        - ds_voxel_size (float, default=1.0): Voxel size used for point cloud downsampling.
        - lx, ux (float): Lower and upper bounds along the x-axis to filter points (default: unbounded).
        - ly, uy (float): Lower and upper bounds along the y-axis to filter points.
        - lz, uz (float): Lower and upper bounds along the z-axis to filter points."""
    
    # Load STL file 
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Sample points uniformly from the surface
    point_cloud = mesh.sample_points_uniformly(number_of_points=pc_number)

    # Convert to np array and remove points out of bounds
    points = np.asarray(point_cloud.points)
    points = points[points[:, 0] < ux]
    points = points[points[:, 0] > lx]
    points = points[points[:, 1] < uy]
    points = points[points[:, 1] > ly]
    points = points[points[:, 2] < uz]
    points = points[points[:, 2] > lz]

    # Create new point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample the point cloud
    voxel_size = ds_voxel_size 
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    print(len(downsampled_points))

    # --- Save to CSV ---
    df = pd.DataFrame(downsampled_points)
    df.to_csv(output_path, index=False)

    return downsampled_pcd

# Input Paths
skin_mesh_path ="/path/to/skin/surface.stl"
skeleton_mesh_path ="/path/to/skeleton/surface.stl"

# Output Paths
skin_csv_path ="/path/to/skin/points.csv"
skeleton_csv_path ="/path/to/skeleton/points.csv"

# Convert
skin_pc = stl_to_csv(skin_mesh_path, skin_csv_path, ux=80, uy=210, lz=35, uz=300)
skeleton_pc = stl_to_csv(skeleton_mesh_path, skeleton_csv_path, ds_voxel_size=2.0, ux=50, lz=50)

# Visualize
skin_pc.paint_uniform_color([1.0, 1.0, 0.0]) 
skeleton_pc.paint_uniform_color([1.0, 0.0, 1.0]) 
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries([skin_pc,skeleton_pc,frame])