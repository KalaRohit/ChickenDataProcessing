import os
import shutil
import sys
import numpy as np
import cv2
import copy
import torch
from lama_cleaner.model.lama import LaMa
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
from typing import List, Tuple
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler

WHITE = [255, 255, 255]
NO_OF_COLORS = 20
COLORS = {}
LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
)


def get_config(strategy):
    data = dict(
        ldm_steps=1,
        ldm_sampler=LDMSampler.plms,
        hd_strategy=strategy,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200,
    )
    return Config(ldm_steps=1,
        ldm_sampler=LDMSampler.plms,
        hd_strategy='HDStrategy',
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=200,
        hd_strategy_resize_limit=200)

def get_cache_path_by_url(url):
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file

def download_model(url):
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file

#Function find_points:
#Gather all the blue points from the image and return its [x,y] coordinates
#param: image -> BGR matrix representing the image.
#output: List of [x,y] where x represents the column and y represents the row.
def find_points(image: np.array) -> List[List[int]]:
    new_lower = np.uint8([[[38, 40, 36]]])

    hsv1 = cv2.cvtColor(new_lower, cv2.COLOR_BGR2HSV)

    print(hsv1)

    image_cpy = copy.deepcopy(image)
    lower_blue = np.array([75,26,40])   
    upper_blue = np.array([145,255,255])

    # lower_blue = np.array([95, 27, 217])
    # upper_blue = np.array([69, 85, 81])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    print('t1', mask.shape)
    # res = cv2.bitwise_not(image_cpy, image_cpy, mask=mask)

    # cv2.imshow("testing", res)
    # cv2.waitKey(0)

    # coords = cv2.findNonZero(mask)  
    # coords = np.array(coords)
    # coords = coords.squeeze(1)
    
    return cv2.inRange(hsv, lower_blue, upper_blue)

#Function blur_points:
#Blurs the point passed in on the image provided.
#Params: image: BGR matrix representation of image, List of two elements x,y 
#        which are the coordinates on the image to be blurred.
#Output: None, does it inplace on the image provided. 
def blur_point(image: np.array, mask) -> None:
    download_model(LAMA_MODEL_URL)
    print(mask.shape)
    device = torch.device(device = 'cuda' if torch.cuda.is_available() else 'cpu')
    model = LaMa(device)
    config = get_config(HDStrategy)
    return model(image, mask, config)


def main() -> None:
    directory = './RawChickenData/'
    images: str = sorted(os.listdir(directory))
    numpy_pic_array: List[np.array] = []
    
    try:
        os.mkdir('./ModifiedChickenData')
    except:
        shutil.rmtree('./ModifiedChickenData')
        os.mkdir('./ModifiedChickenData')
        
    
    for img in images[:1]:
        pic = cv2.imread(directory+img, cv2.IMREAD_COLOR)
        numpy_pic: np.array = np.asarray(pic)
        mask = find_points(numpy_pic)
        blur_point(numpy_pic, mask)
            
    
        numpy_pic_array.append(numpy_pic)

    for index, modified_img in enumerate(numpy_pic_array):
        print(f'writing: {index}.png')
        cv2.imwrite(f'./ModifiedChickenData/{index}.png', modified_img)

    
 
if __name__ == '__main__':
    main()