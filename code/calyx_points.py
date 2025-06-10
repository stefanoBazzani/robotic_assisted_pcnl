import simpleitk as sitk
import numpy as np
from scipy.ndimage import convolve, label
import itk
import matplotlib.pyplot as plt

def skeletonize(volume):
    itk_input = itk.image_from_array(volume.astype(np.uint8))
    itk_input = itk_input.astype(itk.UC)

    # Binary Thinning filter (skeletonization)
    thinning_filter = itk.BinaryThinningImageFilter[itk.Image[itk.UC, 3]].New()
    thinning_filter.SetInput(itk_input)
    thinning_filter.Update()

    skeleton = itk.array_from_image(thinning_filter.GetOutput())
    return skeleton

def find_endpoints(skeleton):
    # Count 3D neighbors using a convolution kernel
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0

    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant')
    endpoints = np.logical_and(skeleton == 1, neighbor_count == 1)
    coords = np.argwhere(endpoints)
    return coords

# Load NIfTI binary mask
input_path = "renal_calyx_seg.nii.gz"
img = sitk.ReadImage(input_path)
volume = sitk.GetArrayFromImage(img)

# Skeletonize volume
skeleton = skeletonize(volume)

# Extract endpoints
endpoints = find_endpoints(skeleton)

print(f"Number of endpoints found: {len(endpoints)}")
for i, pt in enumerate(endpoints):
    print(f"Endpoint {i + 1}: {pt[::-1]} (z, y, x format in NumPy)")

# Save the skeleton as NIfTI for visualization
skeleton_img = sitk.GetImageFromArray(skeleton.astype(np.uint8))
skeleton_img.CopyInformation(img)
sitk.WriteImage(skeleton_img, "renal_calyx_skeleton.nii.gz")
