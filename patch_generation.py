from skimage import filters
from tqdm import trange
from lxml import etree

OPENSLIDE_PATH = r'C:\Users\heeryung\anaconda3\envs\jiwon2\Lib\site-packages\openslide-bin-4.0.0.3-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import numpy as np
import openslide
import argparse
import cv2
import os
import glob

import pandas as pd 


class PatchGenerator:
    def __init__(
        self,
        slide_path,
        anno_path,
        save_path,
        target_patch_size,
        target_mpp,
        tissue_ratio,
        tumor_ratio,
    ):
        self.slide = openslide.open_slide(slide_path)
        self.anno_path = anno_path
        self.save_path = save_path
        self.target_patch_size = target_patch_size
        self.target_mpp = target_mpp
        self.tissue_ratio = tissue_ratio
        self.tumor_ratio = tumor_ratio

        self.min_level = self.slide.level_count - 1
        self.min_downsample = self.slide.level_downsamples[self.slide.level_count - 1]
        self.min_size = self.slide.level_dimensions[self.slide.level_count - 1]
        self.whole_image = np.array(
            self.slide.read_region(
                location=(0, 0), level=self.min_level, size=self.min_size
            )
        )[..., :3]

        # level 0 patch size
        self.level0_mpp = round(float(self.slide.properties.get("openslide.mpp-x")), 2)
        self.level0_patch_size = int((self.target_patch_size * self.target_mpp) / self.level0_mpp)
        self.level0_size = self.slide.level_dimensions[0]

        # level min patch size & mpp
        self.levelm_mpp = self.level0_mpp * self.min_downsample
        self.levelm_patch_size = int((self.target_patch_size * self.target_mpp) / self.levelm_mpp)

    @staticmethod
    def get_tissue_mask(rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        tissue_S = hsv[:, :, 1] > filters.threshold_otsu(hsv[:, :, 1])
        background_R = rgb_image[:, :, 0] > filters.threshold_otsu(rgb_image[:, :, 0])
        background_G = rgb_image[:, :, 1] > filters.threshold_otsu(rgb_image[:, :, 1])
        background_B = rgb_image[:, :, 2] > filters.threshold_otsu(rgb_image[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        mask = tissue_S & (tissue_RGB)
        ret = np.array(mask).astype(np.uint8)
        return ret

    # def get_anno_list(self, anno_path, min_size, min_downsample):
    #     pts_list = []
    #     trees = etree.parse(anno_path).getroot()[0]

    #     for tree in trees:
    #         if tree.get("PartOfGroup") == "Tumor":
    #             regions = tree.findall("Coordinates")
    #             for region in regions:
    #                 coordinates = region.findall("Coordinate")
    #                 pts = list()
    #                 for coord in coordinates:
    #                     x = float(coord.get("X"))
    #                     y = float(coord.get("Y"))
    #                     x = np.clip(round(x / min_downsample), 0, round(min_size[0]))
    #                     y = np.clip(round(y / min_downsample), 0, round(min_size[1]))
    #                     pts.append((x, y))
    #                 pts_list.append(pts)
    #     return pts_list

    def get_anno_list(self, anno_path, min_size, min_downsample):
        pts_list = []
        trees = etree.parse(anno_path).getroot()
        annotations = trees.findall('.//Annotation')

        for annotation in annotations:
            regions = annotation.findall('.//Region')

            for region in regions:
                coordinates = region.findall('.//Vertex')
                pts = list()
                for coord in coordinates:
                        x = float(coord.get('X'))
                        y = float(coord.get('Y'))
                        x = np.clip(round(x/min_downsample), 0, round(min_size[0]))
                        y = np.clip(round(y/min_downsample), 0, round(min_size[1]))
                        pts.append((x,y))
                pts_list.append(pts)
        # if len(pts_list)==1:
        #      pts_list = pts_list[0]
        return pts_list


    def get_anno_mask(self, pts_list, min_size):
        # 주의 numpy로 mask 생성 시 w,h 순서가 아닌 h,w !!
        mask = np.zeros((min_size[1], min_size[0])).astype(np.uint8)
        for pts in pts_list:
            point = [np.array(pts, dtype=np.int32)]
            mask = cv2.fillPoly(mask, point, 1)
        return mask

    def get_ratio_mask(self, patch):
        h_, w_ = patch.shape[0], patch.shape[1]
        n_total = h_ * w_
        n_cell = np.count_nonzero(patch)
        if n_cell != 0:
            return n_cell * 1.0 / n_total * 1.0
        else:
            return 0

    def save_image(self, save_path, patch_name, image_patch):
        os.makedirs(save_path, exist_ok=True)
        image_patch = cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_path, patch_name), image_patch)

    def execute_patch(self, image_patch, target_patch_size, save_path, start_levelm_x, start_levelm_y, patch_count, patch_label):
        resize_image = cv2.resize(image_patch, (target_patch_size, target_patch_size), cv2.INTER_AREA)
        self.save_image(save_path, f"{start_levelm_x}_{start_levelm_y}_{patch_count}_{patch_label}.png", resize_image)

    # Sliding Window
    def sliding_window(self):
        patch_count = 0

        tissue_mask = self.get_tissue_mask(self.whole_image)
        pts_list = self.get_anno_list(self.anno_path, self.min_size, self.min_downsample)
        tumor_mask = self.get_anno_mask(pts_list, self.min_size)

        for start_level0_y in trange(0, self.level0_size[1], self.level0_patch_size):
            for start_level0_x in range(0, self.level0_size[0], self.level0_patch_size):
                start_levelm_x = int(start_level0_x / self.min_downsample)
                start_levelm_y = int(start_level0_y / self.min_downsample)
                end_levelm_x = int((start_level0_x + self.level0_patch_size) / self.min_downsample)
                end_levelm_y = int((start_level0_y + self.level0_patch_size) / self.min_downsample)

                tissue_mask_patch = tissue_mask[start_levelm_y:end_levelm_y, start_levelm_x:end_levelm_x]
                tumor_patch = tumor_mask[start_levelm_y:end_levelm_y, start_levelm_x:end_levelm_x]

                if self.get_ratio_mask(tissue_mask_patch) >= self.tissue_ratio:
                    image_patch = np.array(
                        self.slide.read_region(
                            location=(start_level0_x, start_level0_y),
                            level=0,
                            size=(self.level0_patch_size, self.level0_patch_size),
                        )
                    ).astype(np.uint8)[..., :3]
                    patch_count += 1
                    if self.get_ratio_mask(tumor_patch) >= self.tumor_ratio:
                        self.execute_patch(image_patch, self.target_patch_size, self.save_path, start_level0_x, start_level0_y, patch_count, 1)
                    else:
                        self.execute_patch(image_patch, self.target_patch_size, self.save_path, start_level0_x, start_level0_y, patch_count, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_path", default = 'D:/TCGA_STAD/TCGA_STAD')
    parser.add_argument("--anno_path")
    parser.add_argument("--save_path", default = 'D:/TCGA_STAD/patch_512')
    parser.add_argument("--target_patch_size", type=int, default= 512)
    parser.add_argument("--target_mpp", type=float, default=1.0)
    parser.add_argument("--tissue_ratio", type=float, default=0.5)
    parser.add_argument("--tumor_ratio", type=float, default=0.3)

    args = vars(parser.parse_args())

    slide_path = args["slide_path"]
    anno_path = args["anno_path"]
    save_path = args["save_path"]
    target_patch_size = args["target_patch_size"]
    target_mpp = args["target_mpp"]
    tissue_ratio = args["tissue_ratio"]
    tumor_ratio = args["tumor_ratio"]

    slide_list = []
    anno_list = []
    id_list = []

    ids = os.listdir(slide_path)
    finished_ids = pd.read_csv('D:/TCGA_STAD/her2_class_512_test.csv')['folder_id'].to_list()
    result = [item for item in ids if item not in finished_ids]

    ids_path = [os.path.join(slide_path, id) for id in result]
    print('# of total ids:', len(ids_path))
    slides_path = [''.join(glob.glob(id_path+'/*.svs')) for id_path in ids_path]

    for id_path in ids_path:
        t_dir = os.path.join(id_path, 'T')
        if os.path.exists(t_dir):
            id_list.append(os.path.basename(id_path))
            slide_list.append(''.join(glob.glob(id_path + '/*.svs')))
            anno_list.append(''.join(glob.glob(t_dir + '/*.xml')))
    
    path_dict = {'id': id_list, 'slide': slide_list, 'anno': anno_list}
    # print(path_dict)
    # print(len(path_dict['id']))
    # exit()

    for id, slide, anno in zip(path_dict['id'], path_dict['slide'], path_dict['anno']):
        save = os.path.join(save_path, id)
        os.makedirs(save, exist_ok = True)

        patch_generator = PatchGenerator(
            slide,
            anno,
            save,
            target_patch_size,
            target_mpp,
            tissue_ratio,
            tumor_ratio,
        )
        
        patch_generator.sliding_window()

        print(f'save {save}')
