import os
import math
import numpy as np
import torch
import smplx

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from PIL import Image

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
FRAME_DIR = os.path.join(OUTPUT_DIR, "animation_frames")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)


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
print("faces:", faces.shape)


# ============================================================
# 3. 固定 shape 参数
# 与前面的实验保持一致
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
# 4. 预先计算固定的 v_shaped 和 J
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
# 5. 可视化辅助函数
# ============================================================

def convert_for_visualization(points):
    # SMPL (X, Y, Z)
    # -> display (X, Z, Y)
    return points[:, [0, 2, 1]].copy()


def set_axes_equal(ax, vertices):

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    max_range = max(
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ) / 2.0

    mid_x = (x.max() + x.min()) / 2.0
    mid_y = (y.max() + y.min()) / 2.0
    mid_z = (z.max() + z.min()) / 2.0

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

    ax.set_box_aspect((1, 1, 1))


# ============================================================
# 6. 单帧完整 LBS
# ============================================================

def run_lbs(body_pose):

    global_orient = torch.zeros(
        (1, 3),
        dtype=v_template.dtype,
        device=device
    )

    full_pose = torch.cat(
        [global_orient, body_pose],
        dim=1
    )

    # Axis-angle -> rotation matrices
    rot_mats = batch_rodrigues(
        full_pose.reshape(-1, 3)
    ).reshape(
        1,
        24,
        3,
        3
    )

    # Pose corrective
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

    # Rigid transforms
    J_transformed, A = batch_rigid_transform(
        rot_mats,
        J,
        parents,
        dtype=v_template.dtype
    )

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

    verts = v_homo[
        :,
        :,
        :3,
        0
    ]

    return (
        verts,
        J_transformed
    )


# ============================================================
# 7. 动画参数
# ============================================================

num_frames = 30

# SMPL joint 17 = Right Shoulder
right_shoulder = 17

# body_pose 不含 root joint
right_shoulder_index = right_shoulder - 1

# 从 0 rad 变化到 1.35 rad
start_angle = 0.0
end_angle = 1.35


# ============================================================
# 8. 逐帧生成
# ============================================================

frame_paths = []

for frame_id in range(num_frames):

    t = frame_id / (num_frames - 1)

    # 平滑插值
    smooth_t = 0.5 - 0.5 * math.cos(
        math.pi * t
    )

    angle = (
        start_angle
        +
        (end_angle - start_angle)
        * smooth_t
    )

    body_pose = torch.zeros(
        (1, 69),
        dtype=v_template.dtype,
        device=device
    )

    # 右肩绕 Z 轴逐渐旋转
    body_pose[
        0,
        right_shoulder_index * 3:
        right_shoulder_index * 3 + 3
    ] = torch.tensor(
        [0.0, 0.0, angle],
        dtype=v_template.dtype,
        device=device
    )

    verts, J_transformed = run_lbs(
        body_pose
    )

    verts_np = (
        verts[0]
        .detach()
        .cpu()
        .numpy()
    )

    J_np = (
        J_transformed[0]
        .detach()
        .cpu()
        .numpy()
    )

    verts_vis = convert_for_visualization(
        verts_np
    )

    J_vis = convert_for_visualization(
        J_np
    )

    triangles = verts_vis[faces]

    # ========================================================
    # 绘图
    # ========================================================

    fig = plt.figure(
        figsize=(8, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
        computed_zorder=False
    )

    mesh = Poly3DCollection(
        triangles,
        facecolor=(0.70, 0.78, 0.88, 0.90),
        edgecolor="none",
        linewidths=0.0,
        zorder=1
    )

    ax.add_collection3d(
        mesh
    )

    parents_np = (
        parents
        .detach()
        .cpu()
        .numpy()
    )

    # 骨架
    for joint_id in range(
        1,
        len(parents_np)
    ):

        parent_id = parents_np[joint_id]

        ax.plot(
            [
                J_vis[parent_id, 0],
                J_vis[joint_id, 0]
            ],
            [
                J_vis[parent_id, 1],
                J_vis[joint_id, 1]
            ],
            [
                J_vis[parent_id, 2],
                J_vis[joint_id, 2]
            ],
            color="blue",
            linewidth=2.5,
            zorder=10
        )

    ax.scatter(
        J_vis[:, 0],
        J_vis[:, 1],
        J_vis[:, 2],
        s=35,
        c="red",
        depthshade=False,
        zorder=11
    )

    set_axes_equal(
        ax,
        verts_vis
    )

    ax.view_init(
        elev=0,
        azim=-90
    )

    angle_deg = math.degrees(
        angle
    )

    ax.set_title(
        "Optional Task: SMPL LBS Pose Animation\n"
        f"Right Shoulder Rotation = {angle_deg:.1f}°",
        fontsize=13
    )

    ax.set_xlabel("X")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_zlabel("Height")

    ax.grid(False)

    plt.tight_layout()

    frame_path = os.path.join(
        FRAME_DIR,
        f"frame_{frame_id:03d}.png"
    )

    plt.savefig(
        frame_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    frame_paths.append(
        frame_path
    )

    print(
        f"Frame {frame_id + 1:02d}/{num_frames}: "
        f"{angle_deg:.1f} deg"
    )


# ============================================================
# 9. 合成 GIF
# ============================================================

gif_path = os.path.join(
    OUTPUT_DIR,
    "smpl_lbs_animation.gif"
)

images = [
    Image.open(path).convert("RGB")
    for path in frame_paths
]

images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=80,
    loop=0
)

print()
print("Saved GIF:", gif_path)
print("Optional task finished successfully.")
