import nibabel as nib
import numpy as np
import scipy.ndimage
import csv

def extract_surface_points(nifti_path, output_csv):
    
    # Load volume
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()
    affine = nii.affine  

    # Find Surface Voxels
    structure = np.ones((3, 3, 3))  
    eroded = scipy.ndimage.binary_erosion(volume, structure=structure)  
    surface_mask = volume.astype(bool) & ~eroded  
    surface_voxels = np.argwhere(surface_mask)

    # Coordinate conversion
    surface_real_coords = np.array([nib.affines.apply_affine(affine, voxel) for voxel in surface_voxels])

    # Write Points
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z"])
        writer.writerows(surface_real_coords)

    print(f"Surface file correctly written: {output_csv}")

# Esempio di utilizzo
nifti_file = "/home/ars_control/Desktop/robotic_assisted_pcnl/data/ct/3d_reconstruction/segmentations/WholePhantom_Seg.nii.gz"  # Sostituisci con il tuo file
csv_file = "/home/ars_control/Desktop/robotic_assisted_pcnl/data/ct/3d_reconstruction/segmentations/phantom_surface.csv"
extract_surface_points(nifti_file, csv_file)
