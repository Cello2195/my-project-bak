import os
import cv2
import numpy as np
import glob
import argparse
from utils import get_seg_grayscale, get_densepose, get_cloth_mask_carvekit


def predict(opt):
    # Read input image
    img = cv2.imread("./tmp/origin_web.jpg")
    ori_img = cv2.resize(img, (768, 1024))
    cv2.imwrite("./tmp/origin/origin.jpg", ori_img)

    # Resize input image
    img = cv2.imread('./tmp/origin/origin.jpg')
    img = cv2.resize(img, (384, 512))
    cv2.imwrite('./tmp/resized_img.jpg', img)

    # Get mask of cloth
    print("Get mask of cloth\n")
    get_cloth_mask_carvekit.get_cloth_mask()

    # Get openpose coordinate using openpose
    print("Get openpose coordinate using posenet\n")
    terminal_command = "openpose --image_dir ./tmp/origin/ --hand --disable_blending --display 0 --write_json " \
                       "./repositories/HR-VITON-main/test/test/openpose_json/00001_00_keypoints.json"
    os.system(terminal_command)

    # Generate semantic segmentation using graphonomy-Master library
    print("Generate semantic segmentation using graphonomy-Master library\n")
    terminal_command = "python ./repositories/graphonomy/inference.py --loadmodel  " \
                       "./repositories/graphonomy/inference.pth --img_path ./tmp/resized_img.jpg --output_path ./tmp " \
                       "--output_name resized_segmentation_img"
    os.system(terminal_command)

    # Remove background image using semantic segmentation mask
    mask_img = cv2.imread('tmp/resized_segmentation_img.png', cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (768, 1024))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_img = cv2.erode(mask_img, k)
    img_seg = cv2.bitwise_and(ori_img, ori_img, mask=mask_img)
    back_ground = ori_img - img_seg
    img_seg = np.where(img_seg == 0, 215, img_seg)
    cv2.imwrite("./tmp/seg_img.png", img_seg)
    img = cv2.resize(img_seg, (768, 1024))
    cv2.imwrite('./repositories/HR-VITON-main/test/test/image/00001_00.jpg', img)

    # Generate grayscale semantic segmentation image
    get_seg_grayscale.get_seg_grayscale()

    # Generate Densepose image using detectron2 library
    print("\nGenerate Densepose image using detectron2 library\n")
    terminal_command = "python repositories/detectron2/projects/DensePose/apply_net.py dump " \
                       "repositories/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml " \
                       "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039" \
                       "/model_final_162be9.pkl tmp/origin/origin.jpg --output tmp/output.pkl -v"
    os.system(terminal_command)
    get_densepose.get_densepose()

    # Run HR-VITON to generate final image
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("repositories/HR-VITON-main")
    terminal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth " \
                       "--gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot " \
                       "./test"
    os.system(terminal_command)
    res = glob.glob("./Output/*.png")

    # Add Background or Not
    if opt.background:
        for i in res:
            img = cv2.imread(i)
            img = cv2.bitwise_and(img, img, mask=mask_img)
            img = img + back_ground
            cv2.imwrite(i, img)
    else:
        for i in res:
            img = cv2.imread(i)
            cv2.imwrite(i, img)

    os.chdir("../../")
    cv2.imwrite("./tmp/finalimg.png", img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()
    predict(opt)
