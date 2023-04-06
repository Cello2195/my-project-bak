import cv2
import numpy as np
from PIL import Image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface

def get_cloth_mask():
    SHOW_FULLSIZE = False  # param {type:"boolean"}
    PREPROCESSING_METHOD = "none"  # param ["stub", "none"]
    SEGMENTATION_NETWORK = "tracer_b7"  # param ["u2net", "deeplabv3", "basnet", "tracer_b7"]
    POSTPROCESSING_METHOD = "fba"  # param ["fba", "none"]
    SEGMENTATION_MASK_SIZE = 640  # param ["640", "320"] {type:"raw", allow-input: true}
    TRIMAP_DILATION = 30  # param {type:"integer"}
    TRIMAP_EROSION = 5  # param {type:"integer"}
    DEVICE = 'cpu'  # 'cuda'

    config = MLConfig(segmentation_network=SEGMENTATION_NETWORK,
                      preprocessing_method=PREPROCESSING_METHOD,
                      postprocessing_method=POSTPROCESSING_METHOD,
                      seg_mask_size=SEGMENTATION_MASK_SIZE,
                      trimap_dilation=TRIMAP_DILATION,
                      trimap_erosion=TRIMAP_EROSION,
                      device=DEVICE)

    interface = init_interface(config)
    cloth_img_path = './tmp/cloth_web.jpg'
    imgs = [cloth_img_path]
    images = interface(imgs)
    for i, im in enumerate(images):
        img = np.array(im)
        img = img[..., :3]  # no transparency
        idx = (img[..., 0] == 130) & (img[..., 1] == 130) & (img[..., 2] == 130)  # background 0 or 130, just try it
        img = np.ones(idx.shape) * 255
        img[idx] = 0
        im = Image.fromarray(np.uint8(img), 'L')
        b = cv2.imread(cloth_img_path)
        cv2.imwrite("./repositories/HR-VITON-main/test/test/cloth/00001_00.jpg", b)
        im.save(f'./repositories/HR-VITON-main/test/test/cloth-mask/00001_00.jpg')


if __name__ == '__main__':
    get_cloth_mask()

