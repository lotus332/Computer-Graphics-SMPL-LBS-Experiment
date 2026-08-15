import os
from PIL import Image, ImageOps, ImageDraw, ImageFont

# ============================================================
# 1. 路径设置
# ============================================================

OUTPUT_DIR = "outputs"

image_paths = [
    os.path.join(OUTPUT_DIR, "stage_a_template_weights.png"),
    os.path.join(OUTPUT_DIR, "stage_b_shaped_joints.png"),
    os.path.join(OUTPUT_DIR, "stage_c_pose_offsets.png"),
    os.path.join(OUTPUT_DIR, "stage_d_lbs_result.png"),
]

output_path = os.path.join(
    OUTPUT_DIR,
    "comparison_grid.png"
)


# ============================================================
# 2. 检查文件是否存在
# ============================================================

for path in image_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find image: {path}"
        )

print("All four stage images found.")


# ============================================================
# 3. 读取图片
# ============================================================

images = [
    Image.open(path).convert("RGB")
    for path in image_paths
]


# ============================================================
# 4. 统一每张图尺寸
# ============================================================

# 统一单图尺寸
cell_width = 1000
cell_height = 1000

processed_images = []

for img in images:

    # 保持比例缩放
    img.thumbnail(
        (cell_width, cell_height),
        Image.Resampling.LANCZOS
    )

    # 创建白色背景
    canvas = Image.new(
        "RGB",
        (cell_width, cell_height),
        "white"
    )

    # 居中
    x = (cell_width - img.width) // 2
    y = (cell_height - img.height) // 2

    canvas.paste(
        img,
        (x, y)
    )

    processed_images.append(canvas)


# ============================================================
# 5. 创建 2×2 总画布
# ============================================================

margin = 40
gap = 30

grid_width = (
    margin * 2
    +
    cell_width * 2
    +
    gap
)

grid_height = (
    margin * 2
    +
    cell_height * 2
    +
    gap
)

grid = Image.new(
    "RGB",
    (grid_width, grid_height),
    "white"
)


# ============================================================
# 6. 粘贴四张图
# ============================================================

positions = [
    (margin, margin),
    (
        margin + cell_width + gap,
        margin
    ),
    (
        margin,
        margin + cell_height + gap
    ),
    (
        margin + cell_width + gap,
        margin + cell_height + gap
    )
]

for img, pos in zip(
    processed_images,
    positions
):
    grid.paste(
        img,
        pos
    )


# ============================================================
# 7. 保存
# ============================================================

grid.save(
    output_path,
    quality=95
)

print(
    "Saved:",
    output_path
)

print(
    "Task 6 finished successfully."
)
