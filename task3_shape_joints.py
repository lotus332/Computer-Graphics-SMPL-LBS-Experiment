import os
import numpy as np
import torch
import smplx

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from smplx.lbs import blend_shapes, vertices2joints


# ============================================================
# 1. 路径
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
J_regressor = model.J_regressor.to(device)
faces = model.faces

print("v_template:", tuple(v_template.shape))
print("shapedirs:", tuple(shapedirs.shape))
print("J_regressor:", tuple(J_regressor.shape))


# ============================================================
# 3. 设置非零 shape 参数 beta
# ============================================================

betas = torch.zeros(
    (1, 10),
    dtype=v_template.dtype,
    device=device
)

# 设置前几个非零 shape 参数
# 数值不要过大，否则人体可能变形过于夸张
betas[0, 0] = 2.0
betas[0, 1] = -1.5
betas[0, 2] = 1.0

print()
print("Betas:")
print(betas)


# ============================================================
# 4. 计算 shape blend shapes
# ============================================================

shape_offsets = blend_shapes(
    betas,
    shapedirs
)

print()
print("shape_offsets:", tuple(shape_offsets.shape))


# ============================================================
# 5. 计算 v_shaped
# ============================================================

v_shaped = (
    v_template.unsqueeze(0)
    +
    shape_offsets
)

print("v_shaped:", tuple(v_shaped.shape))


# ============================================================
# 6. 通过 J_regressor 回归关节
# ============================================================

J = vertices2joints(
    J_regressor,
    v_shaped
)

print("J:", tuple(J.shape))


# ============================================================
# 7. 输出形状变化统计
# ============================================================

vertex_displacement = torch.norm(
    v_shaped[0] - v_template,
    dim=1
)

print()
print("Shape displacement statistics:")
print("Mean displacement:",
      vertex_displacement.mean().item())
print("Max displacement:",
      vertex_displacement.max().item())


# ============================================================
# 8. 转 NumPy
# ============================================================

v_template_np = (
    v_template.detach()
    .cpu()
    .numpy()
)

v_shaped_np = (
    v_shaped[0]
    .detach()
    .cpu()
    .numpy()
)

J_np = (
    J[0]
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# 9. 坐标转换，仅用于可视化
# ============================================================

def convert_for_visualization(points):

    # SMPL:
    # X = 左右
    # Y = 上下
    # Z = 前后
    #
    # matplotlib:
    # X = 左右
    # Y = 深度
    # Z = 高度

    return points[:, [0, 2, 1]].copy()


v_shaped_vis = convert_for_visualization(
    v_shaped_np
)

J_vis = convert_for_visualization(
    J_np
)

triangles = v_shaped_vis[faces]


# ============================================================
# 10. 等比例坐标
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
# 11. SMPL 骨骼连接关系
# ============================================================

parents = model.parents.detach().cpu().numpy()

print()
print("parents:")
print(parents)

# ============================================================
# 12. 绘图
# ============================================================

fig = plt.figure(figsize=(9, 10))

# computed_zorder=False：
# 禁止 matplotlib 自动根据3D深度重新排序，
# 这样我们可以手动控制人体、骨骼、关节和文字的显示层级。
ax = fig.add_subplot(
    111,
    projection="3d",
    computed_zorder=False
)


# ============================================================
# 12.1 绘制半透明人体网格
# ============================================================

mesh = Poly3DCollection(
    triangles,
    facecolor=(0.72, 0.72, 0.72, 0.22),
    edgecolor="none",
    linewidths=0.0,
    zorder=1
)

ax.add_collection3d(mesh)


# ============================================================
# 12.2 绘制骨骼
# ============================================================

for joint_id in range(1, len(parents)):

    parent_id = parents[joint_id]

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
        linewidth=3.0,
        alpha=1.0,
        zorder=10
    )


# ============================================================
# 12.3 绘制关节点
# ============================================================

ax.scatter(
    J_vis[:, 0],
    J_vis[:, 1],
    J_vis[:, 2],
    s=55,
    c="red",
    edgecolors="darkred",
    linewidths=0.6,
    depthshade=False,
    zorder=11,
    label="Regressed joints"
)


# ============================================================
# 12.4 绘制关节编号
# ============================================================

for i in range(J_vis.shape[0]):

    # 给文字稍微增加一点横向和竖向偏移，
    # 防止数字正好压在红色关节点上。
    x_offset = 0.012
    z_offset = 0.012

    ax.text(
        J_vis[i, 0] + x_offset,
        J_vis[i, 1],
        J_vis[i, 2] + z_offset,
        str(i),
        fontsize=8,
        color="black",
        fontweight="bold",
        zorder=12,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=0.6
        )
    )


# ============================================================
# 12.5 坐标和视角设置
# ============================================================

set_axes_equal(
    ax,
    v_shaped_vis
)

ax.view_init(
    elev=0,
    azim=-90
)

ax.set_title(
    "(b) Shape Corrected Mesh + Regressed Joints\n"
    r"$v_{shaped}=v_{template}+B_S(\beta)$",
    fontsize=14
)

ax.set_xlabel("X")
ax.set_zlabel("Height")

# 当前是正视图，Depth轴基本没有分析意义，
# 隐藏刻度可以避免左下角大量重叠数字。
ax.set_ylabel("")
ax.set_yticks([])

ax.grid(False)

ax.legend(
    loc="upper right"
)

plt.tight_layout()

# ============================================================
# 13. 保存
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "stage_b_shaped_joints.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Saved:", output_path)
print("Task 3 finished successfully.")
