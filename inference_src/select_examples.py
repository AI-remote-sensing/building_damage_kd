#%%
import os
import cv2

dirs = os.listdir("../data/test/images/")
fn = dirs[0]
# %%
cv2.imread("../data/test/images/" + fn, cv2.IMREAD_COLOR)
# %%
latest_path = "../cls_KD_1610592762_best_best"
dirs = os.listdir(latest_path)
fn = dirs[0]
x = cv2.imread(latest_path + "/" + fn, cv2.IMREAD_COLOR)
x
# %%
latest_path = "../res50cls_cce_0_tuned"
dirs = os.listdir(latest_path)
fn = dirs[0]
x = cv2.imread(latest_path + "/" + fn, cv2.IMREAD_COLOR)
x
# %%
# TODO(sujinhua): find the results why prediction is different from the initial models
latest_path = "../data/test/masks"
dirs = os.listdir(latest_path)
fn = dirs[0]
x = cv2.imread(latest_path + "/" + fn, cv2.IMREAD_COLOR)
x.max()
#%%
# %%
fn = "hurricane-florence_00000004_pre_disaster.png"

# %%
def vis_mask(fn_path, target_folder):
    mask = cv2.imread(fn_path.replace("pre", "post"), cv2.IMREAD_COLOR)
    print(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            label = mask[i, j, 0]
            if label == 0:
                mask[i, j, 0] = 0
                mask[i, j, 1] = 0
                mask[i, j, 2] = 0
                # black
            elif label == 1:
                mask[i, j, 0] = 0
                mask[i, j, 1] = 0
                mask[i, j, 2] = 128
                # blue
            elif label == 2:
                mask[i, j, 0] = 128
                mask[i, j, 1] = 0
                mask[i, j, 2] = 128
                # purple
            elif label == 3:
                mask[i, j, 0] = 255
                mask[i, j, 1] = 255
                mask[i, j, 2] = 0
                # yellow
            elif label == 4:
                mask[i, j, 0] = 255
                mask[i, j, 1] = 0
                mask[i, j, 2] = 0
                # red
    cv2.imwrite(
        target_folder + "/" + fn_path.split("/")[-1].replace(".png", "_vis.png"),
        mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    return mask


vis_mask("../data/test/masks" + "/" + fn, "demo_mask/")
# %%
import numpy as np


def vis_two_part(fn_path, target_folder):
    fn_part1_path = fn_path.replace(".png", "_part1.png.png")
    fn_part2_path = fn_path.replace(".png", "_part2.png.png")
    mask_part1 = cv2.imread(fn_part1_path, cv2.IMREAD_COLOR)
    mask_part2 = cv2.imread(fn_part2_path, cv2.IMREAD_COLOR)
    # print(mask.shape)
    img_array = [
        [0 for j in range(mask_part1.shape[1])] for i in range(mask_part1.shape[0])
    ]
    for i in range(mask_part1.shape[0]):
        for j in range(mask_part1.shape[1]):
            label_list = [
                mask_part1[i, j, 1],
                mask_part1[i, j, 2],
                mask_part2[i, j, 1],
                mask_part2[i, j, 2],
            ]
            label = label_list.index(max(label_list))
            if mask_part1[i, j, 0] < 127:
                img_array[i][j] = (0, 0, 0)  # black
            elif label == 0:
                img_array[i][j] = (0, 0, 128)  # blue
            elif label == 1:
                img_array[i][j] = (128, 0, 128)  # purple
            elif label == 2:
                img_array[i][j] = (255, 255, 0)  # yellow
            elif label == 3:
                img_array[i][j] = (255, 0, 0)  # red
    mask = np.array(img_array)
    cv2.imwrite(
        target_folder + "/" + fn_path.split("/")[-1].replace(".png", "_vis.png"),
        mask,
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    )
    return mask


vis_two_part("../res50cls_cce_0_tuned" + "/" + fn, "demo_init/")
vis_two_part("../cls_KD_1610205999_best_best" + "/" + fn, "demo_kd/")

# %%
