import os
import torch
import smplx

from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform
)


# ============================================================
# 1. 路径
# ============================================================

MODEL_PATH = "/home/lzyuan/Documents/lotusBNUComVision_SMPL/SMPL_NEUTRAL.pkl"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)


# ============================================================
# 2. 加载 SMPL
# ============================================================

model = smplx.create(
    MODEL_PATH,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    num_betas=10,
    batch_size=1
)

device = torch.device("cpu")

model = model.to(device)

v_template = model.v_template.to(device)
shapedirs = model.shapedirs.to(device)
posedirs = model.posedirs.to(device)
J_regressor = model.J_regressor.to(device)
lbs_weights = model.lbs_weights.to(device)
parents = model.parents.to(device)

print("========== Task 7: Manual LBS Verification ==========")

print()
print("Model information:")
print("Vertices:", v_template.shape[0])
print("Faces:", model.faces.shape[0])
print("Joints:", J_regressor.shape[0])
print("Betas dimension:", 10)


# ============================================================
# 3. 与任务 3~5 完全相同的 betas
# ============================================================

betas = torch.zeros(
    (1, 10),
    dtype=v_template.dtype,
    device=device
)

betas[0, 0] = 2.0
betas[0, 1] = -1.5
betas[0, 2] = 1.0


# ============================================================
# 4. 与最终任务5完全相同的姿态
# ============================================================

global_orient = torch.zeros(
    (1, 3),
    dtype=v_template.dtype,
    device=device
)

body_pose = torch.zeros(
    (1, 69),
    dtype=v_template.dtype,
    device=device
)


# ------------------------------------------------------------
# joint IDs
# ------------------------------------------------------------

right_hip = 2

left_shoulder = 16
right_shoulder = 17

left_elbow = 18
right_elbow = 19


right_hip_index = right_hip - 1

left_shoulder_index = left_shoulder - 1
right_shoulder_index = right_shoulder - 1

left_elbow_index = left_elbow - 1
right_elbow_index = right_elbow - 1


# ------------------------------------------------------------
# 左臂
# ------------------------------------------------------------

body_pose[
    0,
    left_shoulder_index * 3:
    left_shoulder_index * 3 + 3
] = torch.tensor(
    [0.0, 0.0, -0.50],
    dtype=v_template.dtype,
    device=device
)

body_pose[
    0,
    left_elbow_index * 3:
    left_elbow_index * 3 + 3
] = torch.tensor(
    [0.0, 0.0, 0.70],
    dtype=v_template.dtype,
    device=device
)


# ------------------------------------------------------------
# 右臂
# ------------------------------------------------------------

body_pose[
    0,
    right_shoulder_index * 3:
    right_shoulder_index * 3 + 3
] = torch.tensor(
    [0.0, 0.0, 1.35],
    dtype=v_template.dtype,
    device=device
)

body_pose[
    0,
    right_elbow_index * 3:
    right_elbow_index * 3 + 3
] = torch.tensor(
    [0.0, 0.0, 0.0],
    dtype=v_template.dtype,
    device=device
)


# ------------------------------------------------------------
# 右腿侧抬
# ------------------------------------------------------------

body_pose[
    0,
    right_hip_index * 3:
    right_hip_index * 3 + 3
] = torch.tensor(
    [0.0, 0.0, -0.70],
    dtype=v_template.dtype,
    device=device
)


print()
print("Betas:")
print(betas)

print()
print("Pose settings:")
print("Left shoulder:",
      body_pose[
          0,
          left_shoulder_index * 3:
          left_shoulder_index * 3 + 3
      ])

print("Left elbow:",
      body_pose[
          0,
          left_elbow_index * 3:
          left_elbow_index * 3 + 3
      ])

print("Right shoulder:",
      body_pose[
          0,
          right_shoulder_index * 3:
          right_shoulder_index * 3 + 3
      ])

print("Right hip:",
      body_pose[
          0,
          right_hip_index * 3:
          right_hip_index * 3 + 3
      ])


# ============================================================
# 5. 手写 LBS：shape
# ============================================================

shape_offsets = blend_shapes(
    betas,
    shapedirs
)

v_shaped = (
    v_template.unsqueeze(0)
    +
    shape_offsets
)

J = vertices2joints(
    J_regressor,
    v_shaped
)


# ============================================================
# 6. 手写 LBS：pose
# ============================================================

full_pose = torch.cat(
    [
        global_orient,
        body_pose
    ],
    dim=1
)

rot_mats = batch_rodrigues(
    full_pose.reshape(-1, 3)
).reshape(
    1,
    24,
    3,
    3
)

ident = torch.eye(
    3,
    dtype=v_template.dtype,
    device=device
)

pose_feature = (
    rot_mats[:, 1:, :, :]
    -
    ident
).reshape(
    1,
    -1
)

pose_offsets = torch.matmul(
    pose_feature,
    posedirs
).reshape(
    1,
    -1,
    3
)

v_posed = (
    v_shaped
    +
    pose_offsets
)


# ============================================================
# 7. 手写 LBS：rigid transform
# ============================================================

J_transformed, A = batch_rigid_transform(
    rot_mats,
    J,
    parents,
    dtype=v_template.dtype
)


# ============================================================
# 8. 手写 LBS：blend transforms
# ============================================================

batch_size = 1
num_vertices = v_template.shape[0]
num_joints = lbs_weights.shape[1]

W = lbs_weights.unsqueeze(0).expand(
    batch_size,
    -1,
    -1
)

A_flat = A.reshape(
    batch_size,
    num_joints,
    16
)

T = torch.matmul(
    W,
    A_flat
).reshape(
    batch_size,
    num_vertices,
    4,
    4
)


# ============================================================
# 9. 手写 LBS：apply transform
# ============================================================

ones = torch.ones(
    (
        batch_size,
        num_vertices,
        1
    ),
    dtype=v_template.dtype,
    device=device
)

v_posed_homo = torch.cat(
    [
        v_posed,
        ones
    ],
    dim=2
)

v_homo = torch.matmul(
    T,
    v_posed_homo.unsqueeze(-1)
)

verts_manual = v_homo[
    :,
    :,
    :3,
    0
]

print()
print("Manual verts shape:",
      tuple(verts_manual.shape))


# ============================================================
# 10. 官方 SMPL forward
# ============================================================

with torch.no_grad():

    output = model(
        betas=betas,
        global_orient=global_orient,
        body_pose=body_pose,
        transl=None,
        return_verts=True
    )

verts_official = output.vertices

print(
    "Official verts shape:",
    tuple(verts_official.shape)
)


# ============================================================
# 11. 逐顶点误差
# ============================================================

abs_error = torch.abs(
    verts_manual
    -
    verts_official
)

mean_absolute_error = (
    abs_error.mean().item()
)

max_absolute_error = (
    abs_error.max().item()
)


# 每个顶点的三维欧氏距离误差
vertex_l2_error = torch.norm(
    verts_manual
    -
    verts_official,
    dim=2
)

mean_vertex_l2 = (
    vertex_l2_error.mean().item()
)

max_vertex_l2 = (
    vertex_l2_error.max().item()
)


print()
print("========== Verification Results ==========")

print(
    "Mean absolute error:",
    mean_absolute_error
)

print(
    "Max absolute error:",
    max_absolute_error
)

print(
    "Mean vertex L2 error:",
    mean_vertex_l2
)

print(
    "Max vertex L2 error:",
    max_vertex_l2
)


# ============================================================
# 12. 判断一致性
# ============================================================

tolerance = 1e-5

passed = (
    max_absolute_error
    <
    tolerance
)

print()
print(
    "Tolerance:",
    tolerance
)

print(
    "Verification passed:",
    passed
)


# ============================================================
# 13. 保存 summary.txt
# ============================================================

summary = f"""
SMPL LBS Experiment Summary
===========================

1. Basic Model Information
---------------------------
Number of vertices: {v_template.shape[0]}
Number of faces: {model.faces.shape[0]}
Number of joints: {J_regressor.shape[0]}
Betas dimension: {betas.shape[1]}

v_template shape: {tuple(v_template.shape)}
shapedirs shape: {tuple(shapedirs.shape)}
posedirs shape: {tuple(posedirs.shape)}
J_regressor shape: {tuple(J_regressor.shape)}
lbs_weights shape: {tuple(lbs_weights.shape)}

2. Shape Parameters
-------------------
betas:
{betas.detach().cpu().numpy()}

3. Pose Parameters
------------------
Left shoulder axis-angle:
{body_pose[0, left_shoulder_index*3:left_shoulder_index*3+3].detach().cpu().numpy()}

Left elbow axis-angle:
{body_pose[0, left_elbow_index*3:left_elbow_index*3+3].detach().cpu().numpy()}

Right shoulder axis-angle:
{body_pose[0, right_shoulder_index*3:right_shoulder_index*3+3].detach().cpu().numpy()}

Right elbow axis-angle:
{body_pose[0, right_elbow_index*3:right_elbow_index*3+3].detach().cpu().numpy()}

Right hip axis-angle:
{body_pose[0, right_hip_index*3:right_hip_index*3+3].detach().cpu().numpy()}

4. Manual LBS Intermediate Shapes
---------------------------------
v_shaped: {tuple(v_shaped.shape)}
J: {tuple(J.shape)}
rot_mats: {tuple(rot_mats.shape)}
pose_feature: {tuple(pose_feature.shape)}
pose_offsets: {tuple(pose_offsets.shape)}
v_posed: {tuple(v_posed.shape)}
J_transformed: {tuple(J_transformed.shape)}
A: {tuple(A.shape)}
W: {tuple(W.shape)}
T: {tuple(T.shape)}
verts_manual: {tuple(verts_manual.shape)}

5. Manual LBS vs Official SMPL
------------------------------
Mean absolute error: {mean_absolute_error:.12e}
Max absolute error: {max_absolute_error:.12e}

Mean vertex L2 error: {mean_vertex_l2:.12e}
Max vertex L2 error: {max_vertex_l2:.12e}

Tolerance: {tolerance:.1e}
Verification passed: {passed}

Conclusion
----------
The manually implemented Linear Blend Skinning result was compared
vertex-by-vertex with the official SMPL forward result using exactly
the same betas, global orientation, and body pose parameters.
"""

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)

print()
print(
    "Saved:",
    SUMMARY_PATH
)

print(
    "Task 7 finished successfully."
)
