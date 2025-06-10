## CT Files

### CT Volume
The original CT volume of the mannequin is stored within *phantom_ct_volume.nii.gz* file.

### 3D Slicer Scene
[3d_reconstruction](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/data/ct/3d_reconstruction) folder contains the files of a 3D Slicer scene.
Follow these step to visualize the scene:
- Download and install *3DSlicer* from [this link](https://download.slicer.org/)
- Download the [3d_reconstruction](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/data/ct/3d_reconstruction) folder
- Drag and Drop *phantom_scene.mrml* file into 3DSlicer

![3DSlicer Scene](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/data/ct/3d_reconstruction/phantom_scene.png)

### Segmentations
3D segmented organs are stored into [segmenttions](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/data/ct/segmentations) as .stl surface files.
These file are converted into .csv points lists using the *script volume_pre_processing.py* stored in [this folder](https://github.com/stefanoBazzani/robotic_assisted_pcnl/tree/main/code)
