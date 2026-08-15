# Computer Graphics Experiment 8: Linear Blend Skinning

This repository contains the implementation of Experiment 8 for the Computer Graphics course.

## Experiment

Linear Blend Skinning (LBS) based on the SMPL human body model.

The implementation explicitly extracts and visualizes the main intermediate variables of the SMPL LBS pipeline:

- `v_template`: template mesh
- `v_shaped`: shape-corrected mesh
- `J`: regressed joints
- `v_posed`: pose-corrected mesh
- `verts`: final LBS vertices

## Pipeline

The experiment contains the following stages:

1. Load the SMPL model and print basic model information.
2. Visualize the template mesh and skinning weights.
3. Apply shape blend shapes and regress the joints.
4. Compute pose corrective offsets.
5. Implement the complete Linear Blend Skinning procedure.
6. Generate a four-stage comparison image.
7. Compare the manually implemented LBS result with the official SMPL forward result.

## Files

```text
task1_load_smpl.py
task2_weights.py
task3_shape_joints.py
task4_pose_offsets.py
task5_lbs_result.py
task6_comparison.py
task7_verify_lbs.py
