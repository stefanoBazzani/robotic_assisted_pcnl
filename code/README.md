# Scripts

## Volume pre-processing
Organs and anatomical structures can be segmented using either manual/traditional techniques or AI-based methods. In our workflow, we utilize [nnUNet](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) for segmenting kidneys and stones, while thresholding-based approaches are applied for segmenting bones and skin.

For pre-operative needle trajectory planning, the skin surface and skeleton point clouds are essential. After segmentation, these structures are saved as .stl files and then processed using the '''volumes_pre_processing.py''' script, which converts the 3D volume meshes into .csv point clouds suitable for further analysis.

## Optimal trajectory planning
The '''optimize_trajectory_30.py''' script computes an optimal needle trajectory with a fixed 30° inclination. The required inputs are:
- Skin surface point cloud (.csv)
- Skeleton point cloud (.csv)
- Desired calyx insertion point and preferred needle direction

The script outputs the optimal probe pose (position and orientation) ensuring the needle, inserted via the robot-mounted needle guide, follows the planned trajectory while respecting anatomical constraints such as skin tangency, calyx targeting, and collision avoidance with internal organs.

