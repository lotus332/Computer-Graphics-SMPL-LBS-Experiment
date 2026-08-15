import os
import numpy as np
import torch
import smplx

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues
)


# ============================================================
# 1. 路径设置
# ============================================================

MODEL_PATH = "/home/lzyuan/Documents/lotusBNUComVision_SMPL/SMPL_NEUTRAL.pkl"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

v_template = model.v_template.to(device)
shapedirs = model.shapedirs.to(device)
posedirs = model.posedirs.to(device)
J_regressor = model.J_regressor.to(device)
faces = model.faces

print("v_template:", tuple(v_template.shape))
print("shapedirs:", tuple(shapedirs.shape))
print("posedirs:", tuple(posedirs.shape))
print("J_regressor:", tuple(J_regressor.shape))


# ============================================================
# 3. 与任务3、任务5保持一致的 shape 参数
# ============================================================

betas = torch.zeros(
    (1, 10),
    dtype=v_template.dtype,
    device=device
)

betas[0, 0] = 2.0
betas[0, 1] = -1.5
betas[0, 2] = 1.0

print()
print("Betas:")
print(betas)


# ============================================================
# 4. 计算 v_shaped
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

print()
print("shape_offsets:", tuple(shape_offsets.shape))
print("v_shaped:", tuple(v_shaped.shape))
print("J:", tuple(J.shape))


# ============================================================
# 5. 设置与任务5完全一致的姿态
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
# SMPL joint IDs
# ------------------------------------------------------------

right_hip = 2

left_shoulder = 16
right_shoulder = 17

left_elbow = 18
right_elbow = 19


# body_pose 中 joint k 对应 k - 1
right_hip_index = right_hip - 1

left_shoulder_index = left_shoulder - 1
right_shoulder_index = right_shoulder - 1

left_elbow_index = left_elbow - 1
right_elbow_index = right_elbow - 1


# ============================================================
# 5.1 左臂：保持最终姿势
# ============================================================

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


# ============================================================
# 5.2 右臂：自然下垂
# ============================================================

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


# ============================================================
# 5.3 右腿：向侧面抬起
# ============================================================

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
print("Pose settings:")

print(
    "Left shoulder:",
    body_pose[
        0,
        left_shoulder_index * 3:
        left_shoulder_index * 3 + 3
    ]
)

print(
    "Left elbow:",
    body_pose[
        0,
        left_elbow_index * 3:
        left_elbow_index * 3 + 3
    ]
)

print(
    "Right shoulder:",
    body_pose[
        0,
        right_shoulder_index * 3:
        right_shoulder_index * 3 + 3
    ]
)

print(
    "Right elbow:",
    body_pose[
        0,
        right_elbow_index * 3:
        right_elbow_index * 3 + 3
    ]
)

print(
    "Right hip:",
    body_pose[
        0,
        right_hip_index * 3:
        right_hip_index * 3 + 3
    ]
)


# ============================================================
# 6. 拼接完整姿态
# ============================================================

full_pose = torch.cat(
    [global_orient, body_pose],
    dim=1
)

print()
print("full_pose:", tuple(full_pose.shape))


# ============================================================
# 7. Axis-angle -> rotation matrices
# ============================================================

rot_mats = batch_rodrigues(
    full_pose.reshape(-1, 3)
).reshape(
    1,
    24,
    3,
    3
)

print("rot_mats:", tuple(rot_mats.shape))


# ============================================================
# 8. 构造 pose_feature = R - I
# ============================================================

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

print("pose_feature:", tuple(pose_feature.shape))


# ============================================================
# 9. 计算 pose_offsets
# ============================================================

pose_offsets = torch.matmul(
    pose_feature,
    posedirs
).reshape(
    1,
    -1,
    3
)

print("pose_offsets:", tuple(pose_offsets.shape))


# ============================================================
# 10. 计算 v_posed
# ============================================================

v_posed = (
    v_shaped
    +
    pose_offsets
)

print("v_posed:", tuple(v_posed.shape))


# ============================================================
# 11. 计算 pose offset 大小
# ============================================================

pose_offset_magnitude = torch.norm(
    pose_offsets[0],
    dim=1
)

print()
print("Pose offset statistics:")

print(
    "Mean magnitude:",
    pose_offset_magnitude.mean().item()
)

print(
    "Max magnitude:",
    pose_offset_magnitude.max().item()
)

print(
    "Min magnitude:",
    pose_offset_magnitude.min().item()
)

max_vertex = torch.argmax(
    pose_offset_magnitude
).item()

print(
    "Vertex with maximum pose offset:",
    max_vertex
)


# ============================================================
# 12. 转 NumPy
# ============================================================

v_posed_np = (
    v_posed[0]
    .detach()
    .cpu()
    .numpy()
)

pose_magnitude_np = (
    pose_offset_magnitude
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# 13. 坐标转换，仅用于可视化
# ============================================================

def convert_for_visualization(points):

    # SMPL:
    # X = 左右
    # Y = 高度
    # Z = 前后
    #
    # 显示：
    # X = 左右
    # Y = 深度
    # Z = 高度

    return points[:, [0, 2, 1]].copy()


v_posed_vis = convert_for_visualization(
    v_posed_np
)

triangles = v_posed_vis[faces]


# ============================================================
# 14. 等比例坐标
# ============================================================

def set_axes_equal(ax, vertices):

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2.0

    mid_x = (
        x.max() + x.min()
    ) / 2.0

    mid_y = (
        y.max() + y.min()
    ) / 2.0

    mid_z = (
        z.max() + z.min()
    ) / 2.0

    ax.set_xlim(
        mid_x - max_range,
        mid_x + max_range
    )

    ax.set_ylim(
        mid_y - max_range,
        mid_y + max_range
    )

    ax.set_zlim(
        mid_z - max_range,
        mid_z + max_range
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )


# ============================================================
# 15. 根据 pose offset 大小设置热力颜色
# ============================================================

face_magnitude = (
    pose_magnitude_np[faces]
    .mean(axis=1)
)

# 使用99百分位增强颜色区分
vmax = np.percentile(
    pose_magnitude_np,
    99
)

if vmax <= 0:
    vmax = pose_magnitude_np.max()

norm = Normalize(
    vmin=0.0,
    vmax=vmax
)

cmap = plt.get_cmap(
    "turbo"
)

face_colors = cmap(
    norm(face_magnitude)
)


# ============================================================
# 16. 绘制 Pose Corrective Offset 热力图
# ============================================================

fig = plt.figure(
    figsize=(9, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

mesh = Poly3DCollection(
    triangles,
    facecolors=face_colors,
    edgecolor="none",
    linewidths=0.01
)

ax.add_collection3d(
    mesh
)


set_axes_equal(
    ax,
    v_posed_vis
)

ax.view_init(
    elev=0,
    azim=-90
)

ax.set_title(
    "(c) Pose Corrective Offsets\n"
    r"$v_{posed}=v_{shaped}+B_P(\theta)$",
    fontsize=14
)

ax.set_xlabel("X")
ax.set_ylabel("")
ax.set_yticks([])
ax.set_zlabel("Height")

ax.grid(False)


# ============================================================
# 17. Colorbar
# ============================================================

sm = cm.ScalarMappable(
    norm=norm,
    cmap=cmap
)

sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    fraction=0.03,
    pad=0.05
)

cbar.set_label(
    "Pose Offset Magnitude"
)

plt.tight_layout()


# ============================================================
# 18. 保存
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "stage_c_pose_offsets.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:", output_path)
print("Task 4 finished successfully.")
