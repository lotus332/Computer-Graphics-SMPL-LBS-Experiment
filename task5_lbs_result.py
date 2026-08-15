import os
import numpy as np
import torch
import smplx

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform
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
lbs_weights = model.lbs_weights.to(device)
parents = model.parents.to(device)
faces = model.faces

print("v_template:", tuple(v_template.shape))
print("shapedirs:", tuple(shapedirs.shape))
print("posedirs:", tuple(posedirs.shape))
print("J_regressor:", tuple(J_regressor.shape))
print("lbs_weights:", tuple(lbs_weights.shape))
print("parents:", tuple(parents.shape))


# ============================================================
# 3. Shape 参数
# 与任务3、4保持一致
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
# 4. 计算 v_shaped 与 J
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
# 5. 设置姿态参数
# ============================================================

# root / pelvis
global_orient = torch.zeros(
    (1, 3),
    dtype=v_template.dtype,
    device=device
)

# 剩余23个关节，每个3维 axis-angle
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


# body_pose 不包括 joint 0
# joint k 在 body_pose 中对应索引 k - 1

right_hip_index = right_hip - 1

left_shoulder_index = left_shoulder - 1
right_shoulder_index = right_shoulder - 1

left_elbow_index = left_elbow - 1
right_elbow_index = right_elbow - 1


# ============================================================
# 5.1 左臂：保持当前姿势
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

# 从 T-pose 向身体下方旋转
body_pose[
    0,
    right_shoulder_index * 3:
    right_shoulder_index * 3 + 3
] = torch.tensor(
    [0.0, 0.0, 1.35],
    dtype=v_template.dtype,
    device=device
)

# 右肘基本伸直
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
# 5.3 右腿：向身体右侧抬起
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
# 6. 拼接完整姿态 full_pose
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
# 8. 计算 pose corrective
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

print("pose_feature:", tuple(pose_feature.shape))
print("pose_offsets:", tuple(pose_offsets.shape))
print("v_posed:", tuple(v_posed.shape))


# ============================================================
# 9. 计算运动学链上的刚体变换
# ============================================================

J_transformed, A = batch_rigid_transform(
    rot_mats,
    J,
    parents,
    dtype=v_template.dtype
)

print()
print("J_transformed:", tuple(J_transformed.shape))
print("A:", tuple(A.shape))


# ============================================================
# 10. 计算每个顶点的混合变换矩阵
# ============================================================

batch_size = 1
num_vertices = v_template.shape[0]
num_joints = lbs_weights.shape[1]

# (6890, 24)
# -> (1, 6890, 24)
W = lbs_weights.unsqueeze(0).expand(
    batch_size,
    -1,
    -1
)

print("W:", tuple(W.shape))


# (1, 24, 4, 4)
# -> (1, 24, 16)
A_flat = A.reshape(
    batch_size,
    num_joints,
    16
)

# 对24个关节变换进行权重混合
T = torch.matmul(
    W,
    A_flat
).reshape(
    batch_size,
    num_vertices,
    4,
    4
)

print("T:", tuple(T.shape))


# ============================================================
# 11. v_posed -> 齐次坐标
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

print("v_posed_homo:", tuple(v_posed_homo.shape))


# ============================================================
# 12. 最终 Linear Blend Skinning
# ============================================================

v_homo = torch.matmul(
    T,
    v_posed_homo.unsqueeze(-1)
)

verts = v_homo[
    :,
    :,
    :3,
    0
]

print("v_homo:", tuple(v_homo.shape))
print("verts:", tuple(verts.shape))


# ============================================================
# 13. LBS 位移统计
# ============================================================

lbs_displacement = torch.norm(
    verts[0] - v_posed[0],
    dim=1
)

print()
print("LBS displacement statistics:")

print(
    "Mean displacement:",
    lbs_displacement.mean().item()
)

print(
    "Max displacement:",
    lbs_displacement.max().item()
)


# ============================================================
# 14. 转 NumPy
# ============================================================

verts_np = (
    verts[0]
    .detach()
    .cpu()
    .numpy()
)

J_transformed_np = (
    J_transformed[0]
    .detach()
    .cpu()
    .numpy()
)

parents_np = (
    parents
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# 15. 仅用于可视化的坐标转换
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


verts_vis = convert_for_visualization(
    verts_np
)

J_transformed_vis = convert_for_visualization(
    J_transformed_np
)

triangles = verts_vis[faces]


# ============================================================
# 16. 等比例显示
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
# 17. 绘制最终 LBS 人体
# ============================================================

fig = plt.figure(
    figsize=(9, 10)
)

ax = fig.add_subplot(
    111,
    projection="3d",
    computed_zorder=False
)


# ------------------------------------------------------------
# 最终人体网格
# ------------------------------------------------------------

mesh = Poly3DCollection(
    triangles,
    facecolor=(0.70, 0.78, 0.88, 0.85),
    edgecolor="none",
    linewidths=0.0,
    zorder=1
)

ax.add_collection3d(
    mesh
)


# ------------------------------------------------------------
# 最终骨架
# ------------------------------------------------------------

for joint_id in range(
    1,
    len(parents_np)
):

    parent_id = parents_np[joint_id]

    ax.plot(
        [
            J_transformed_vis[parent_id, 0],
            J_transformed_vis[joint_id, 0]
        ],
        [
            J_transformed_vis[parent_id, 1],
            J_transformed_vis[joint_id, 1]
        ],
        [
            J_transformed_vis[parent_id, 2],
            J_transformed_vis[joint_id, 2]
        ],
        color="blue",
        linewidth=3.0,
        zorder=10
    )


# ------------------------------------------------------------
# 最终关节点
# ------------------------------------------------------------

ax.scatter(
    J_transformed_vis[:, 0],
    J_transformed_vis[:, 1],
    J_transformed_vis[:, 2],
    s=50,
    c="red",
    edgecolors="darkred",
    linewidths=0.6,
    depthshade=False,
    zorder=11,
    label="Transformed joints"
)


# ------------------------------------------------------------
# 标出关节编号
# ------------------------------------------------------------

for i in range(
    J_transformed_vis.shape[0]
):

    ax.text(
        J_transformed_vis[i, 0] + 0.012,
        J_transformed_vis[i, 1],
        J_transformed_vis[i, 2] + 0.012,
        str(i),
        fontsize=8,
        color="black",
        fontweight="bold",
        zorder=12,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.70,
            pad=0.5
        )
    )


# ============================================================
# 18. 视角设置
# ============================================================

set_axes_equal(
    ax,
    verts_vis
)

ax.view_init(
    elev=0,
    azim=-90
)

ax.set_title(
    "(d) Final LBS Result\n"
    r"$v_i'=\sum_k w_{ik}G_kv_i^{posed}$",
    fontsize=14
)

ax.set_xlabel("X")
ax.set_ylabel("")
ax.set_yticks([])
ax.set_zlabel("Height")

ax.grid(False)

ax.legend(
    loc="upper right"
)

plt.tight_layout()


# ============================================================
# 19. 保存
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "stage_d_lbs_result.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:", output_path)
print("Task 5 finished successfully.")
