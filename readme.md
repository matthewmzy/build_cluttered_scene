# A project for scene building, visualizing and validating grasp poses

## hierarchy
- data
  - dataset : grasp pose data from dexgraspnet after validation
  - graspdata : grasp pose data from dexgraspnet before validation, currently include 50 grasps for each object
  - meshdata : object mesh and urdf
  - scene_data : restore object codes and their translations and rotations
  - visualized_data : scene_data after the screen by visualize_scene.py
  - result : log of validating scene level grasp
- mjcf : hand assets
- open_ai_assets : hand assets
- scripts : introduced in ppt
  - build_scene.py
  - validate_scene.py
  - visualize_scene.py
- utils : hand model loading and rotation processing utils

## to use
```bash
python isaacgym_simu/scripts/build_scene.py --result_file scenes_2.npy
python isaacgym_simu/scripts/visualize_scene.py --data_file scenes_2.npy --save_file scene_2_.npy --save_result
cd isaacgym_simu
python scripts/validate_scene.py --visualize --save_result --scene_file scenes_2_.npy
```