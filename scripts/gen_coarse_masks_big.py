# -----------------------------------------------------------------------------------
# References:
# cascadepsp: https://github.com/hkchengrex/CascadePSP/blob/83cc3b8783b595b2e47c75016f93654eaddb7412/util/boundary_modification.py
# -----------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import random
import math
from tqdm import tqdm
from PIL import Image


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()

    _, seg = cv2.threshold(seg, 127, 255, 0)


    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg


    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)


            if np.random.rand() < 0.25:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])

        if compute_iou(seg, gt) < iou_target:
            break

    return seg

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target = 0.8):










    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


    sampled_contours = []
    modified_contours = []

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)


        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)

        idx_dist = []
        for i in range(number_of_vertices-number_of_removes):
            idx_dist.append([i, np.sum((contour[i] - contour[i+number_of_removes])**2)])

        idx_dist = sorted(idx_dist, key=lambda x:x[1])

        remove_start = random.choice(idx_dist[:math.ceil(0.1*len(idx_dist))])[0]


        new_contour = np.concatenate([contour[:remove_start], contour[remove_start+number_of_removes:]], axis=0)
        contour = new_contour



        number_of_vertices = contour.shape[0]
        indices = random.sample(range(number_of_vertices), int(number_of_vertices * sample_rate))
        indices.sort()
        sampled_contour = contour[indices]
        sampled_contours.append(sampled_contour)

        modified_contour = np.copy(sampled_contour)
        if (M['m00'] != 0):
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])


            for idx, coor in enumerate(modified_contour):

                change = np.random.normal(0,move_rate)
                x,y = coor[0]
                new_x = x + (x-center[0]) * change
                new_y = y + (y-center[1]) * change

                modified_contour[idx] = [new_x,new_y]
        modified_contours.append(modified_contour)


    gt = np.copy(image)
    image = np.zeros((image.shape[0], image.shape[1], 3))

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        image = gt.copy()
    else:
        image = cv2.drawContours(image, modified_contours, -1, (255, 0, 0), -1)

    if len(image.shape) == 3:
        image = image[:, :, 0]
    image = perturb_seg(image, iou_target)


    image = (image >= 128).astype(np.uint8) * 255

    return image


if __name__ == '__main__':
    data_root = r'F:\syc\lunwen3\1_gaijin\SegRefiner\data\BIG\test'

    print('----------start generating coarse masks for BIG dataset---------------')


    all_files = os.listdir(data_root)
    gt_files = [f for f in all_files if f.endswith('_gt.png')]

    print(f'Found {len(gt_files)} GT files')

    with tqdm(total=len(gt_files)) as p:
        for gt_file in gt_files:

            base_name = gt_file.replace('_gt.png', '')


            gt_path = os.path.join(data_root, gt_file)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            if gt_mask is None:
                print(f'Warning: Could not read {gt_path}')
                continue


            coarse_mask = modify_boundary(gt_mask)


            coarse_filename = os.path.join(data_root, base_name + '_seg.png')
            Image.fromarray(coarse_mask).save(coarse_filename)

            p.update()

    print('----------coarse mask generation completed---------------')