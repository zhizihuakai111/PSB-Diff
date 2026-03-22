# -----------------------------------------------------------------------------------
# References:
# cascadepsp: https://github.com/hkchengrex/CascadePSP/blob/83cc3b8783b595b2e47c75016f93654eaddb7412/util/boundary_modification.py
# -----------------------------------------------------------------------------------
import os
import cv2
import glob
import numpy as np
import random
import math
import json
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image

DATA_ROOT = '/home/ubt1234/syc/1_lunwen3/3_R1/SegRefiner/data'


try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

CAND_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')



SMALL_AREA = 50

def get_random_structure(size):

    choice = np.random.randint(1, 5)
    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size // 2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    return cv2.dilate(seg, kernel, iterations=1)

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    return cv2.erode(seg, kernel, iterations=1)

def compute_iou(seg, gt):
    intersection = seg * gt
    union = seg + gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)

def perturb_seg(gt, iou_target=0.6):
    h, w = gt.shape
    seg = gt.copy()
    _, seg = cv2.threshold(seg, 127, 255, 0)

    if h <= 2 or w <= 2:

        return seg


    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx + 1, w + 1), np.random.randint(ly + 1, h + 1)

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

def modify_boundary(image, regional_sample_rate=0.1, sample_rate=0.1, move_rate=0.0, iou_target=0.8):

    if int(cv2.__version__[0]) >= 4:
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    modified_contours = []

    for contour in contours:
        if contour.shape[0] < 10:
            continue
        M = cv2.moments(contour)


        number_of_vertices = contour.shape[0]
        number_of_removes = int(number_of_vertices * regional_sample_rate)

        if number_of_removes > 0 and number_of_vertices - number_of_removes > 1:
            idx_dist = []
            for i in range(number_of_vertices - number_of_removes):
                idx_dist.append([i, np.sum((contour[i] - contour[i + number_of_removes]) ** 2)])
            idx_dist = sorted(idx_dist, key=lambda x: x[1])
            topk = max(1, math.ceil(0.1 * len(idx_dist)))
            remove_start = random.choice(idx_dist[:topk])[0]
            contour = np.concatenate([contour[:remove_start], contour[remove_start + number_of_removes:]], axis=0)


        number_of_vertices = contour.shape[0]
        keep_n = max(1, int(number_of_vertices * sample_rate))
        indices = sorted(random.sample(range(number_of_vertices), keep_n))
        sampled_contour = contour[indices]


        modified_contour = np.copy(sampled_contour)
        if M['m00'] != 0:
            center = round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])
            for idx, coor in enumerate(modified_contour):
                change = np.random.normal(0, move_rate)
                x, y = coor[0]
                new_x = x + (x - center[0]) * change
                new_y = y + (y - center[1]) * change
                modified_contour[idx] = [new_x, new_y]
        modified_contours.append(modified_contour)


    gt = np.copy(image)
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    modified_contours = [cont for cont in modified_contours if len(cont) > 0]
    if len(modified_contours) == 0:
        canvas = gt.copy()
    else:
        canvas = cv2.drawContours(canvas, modified_contours, -1, (255, 0, 0), -1)

    if len(canvas.shape) == 3:
        canvas = canvas[:, :, 0]
    canvas = perturb_seg(canvas, iou_target)
    canvas = (canvas >= 128).astype(np.uint8) * 255
    return canvas

def try_find_image_path(base_path_wo_ext):
    """不读图，先检查文件是否存在；优先常见扩展名，然后 glob 兜底"""
    for ext in CAND_EXTS:
        p = base_path_wo_ext + ext
        if os.path.isfile(p):
            return p
    globs = glob.glob(base_path_wo_ext + '.*')
    if globs:
        return globs[0]
    return None

def try_imread_with_exts(base_path_wo_ext):
    """返回 (img, abs_path)。如果找不到或读不到，返回 (None, None)。"""
    p = try_find_image_path(base_path_wo_ext)
    if p is None:
        return None, None
    img = cv2.imread(p)
    if img is None:
        return None, None
    return img, p

def run_inst(img_info):
    gt_mask_path = os.path.join(DATA_ROOT, img_info['maskname'])
    coarse_path  = os.path.join(DATA_ROOT, img_info['coarsename'])
    expand_path  = os.path.join(DATA_ROOT, img_info['expandname'])

    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"[WARN] Mask not found: {gt_mask_path}")
        return


    gt_bin = (gt_mask >= 128).astype(np.uint8)
    area = int(gt_bin.sum())

    if area < SMALL_AREA:

        Image.fromarray((gt_bin * 255).astype('uint8')).save(coarse_path)
        Image.fromarray((gt_bin * 255).astype('uint8')).save(expand_path)
        return


    coarse_mask = modify_boundary(gt_bin * 255)
    Image.fromarray(coarse_mask).save(coarse_path)


    contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expand_coarse_mask = np.zeros_like(gt_bin)
    cv2.drawContours(expand_coarse_mask, contours, contourIdx=-1, color=1, thickness=-1)
    expand_coarse_mask = modify_boundary(expand_coarse_mask * 255)
    Image.fromarray(expand_coarse_mask).save(expand_path)

if __name__ == '__main__':



    THIN_TXT  = os.path.join(DATA_ROOT, 'thin_object', 'list', 'train.txt')
    THIN_IMG_DIR = os.path.join(DATA_ROOT, 'thin_object', 'images')
    THIN_MASK_DIR = os.path.join(DATA_ROOT, 'thin_object', 'masks')
    OUT_COARSE_DIR = os.path.join(DATA_ROOT, 'thin_object', 'coarse')
    OUT_EXPAND_DIR = os.path.join(DATA_ROOT, 'thin_object', 'coarse_expand')
    COLLECTION_JSON = os.path.join(DATA_ROOT, 'collection_hr.json')

    os.makedirs(OUT_COARSE_DIR, exist_ok=True)
    os.makedirs(OUT_EXPAND_DIR, exist_ok=True)

    collection = dict(thin=[])

    print('----------start collecting thin---------------')
    with open(THIN_TXT, 'r', encoding='utf-8') as f:
        raw_names = [line.strip() for line in f if line.strip()]


    base_names = [os.path.splitext(n)[0] for n in raw_names]

    valid_items = []
    missing_images, missing_masks = [], []

    with tqdm(total=len(base_names)) as pbar:
        for base in base_names:

            mask_abs = os.path.join(THIN_MASK_DIR, base + '.png')
            mask_rel = os.path.join('thin_object', 'masks', base + '.png')

            if not os.path.isfile(mask_abs):
                missing_masks.append(mask_abs)
                pbar.update()
                continue


            img_base_no_ext = os.path.join(THIN_IMG_DIR, base)
            img, img_abs = try_imread_with_exts(img_base_no_ext)
            if img is None:
                missing_images.append(img_base_no_ext)
                pbar.update()
                continue

            h, w = img.shape[:2]
            img_rel = os.path.join('thin_object', 'images', os.path.basename(img_abs))
            coarse_rel = os.path.join('thin_object', 'coarse', base + '.png')
            expand_rel = os.path.join('thin_object', 'coarse_expand', base + '.png')

            item = {
                'filename': img_rel,
                'maskname': mask_rel,
                'coarsename': coarse_rel,
                'expandname': expand_rel,
                'height': h, 'width': w
            }
            valid_items.append(item)
            collection['thin'].append(item)
            pbar.update()

    print(f'Collected {len(valid_items)} / {len(base_names)} items.')
    if missing_images or missing_masks:
        print(f'Missing images: {len(missing_images)}, Missing masks: {len(missing_masks)}')
        if missing_images:
            with open(os.path.join(DATA_ROOT, 'missing_images.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(missing_images))
        if missing_masks:
            with open(os.path.join(DATA_ROOT, 'missing_masks.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(missing_masks))

    print('----------start transforming thin---------------')
    procs = min(20, max(1, mp.cpu_count() - 1))
    with mp.Pool(processes=procs) as pool:
        with tqdm(total=len(valid_items)) as pbar:
            for _ in pool.imap_unordered(run_inst, valid_items):
                pbar.update()

    print('----------writing json file---------------')
    with open(COLLECTION_JSON, 'w', encoding='utf-8') as f:
        json.dump(collection, f, ensure_ascii=False, indent=2)

    print('Done.')