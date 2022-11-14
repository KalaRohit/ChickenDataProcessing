import os
import shutil
import sys
import numpy as np

from PIL import Image
from typing import List, Tuple

WHITE = [255, 255, 255] 

#read the csv file for the points. Return as (x,y) 
def find_points(image: np.array) -> List[Tuple[int]]:
    return []

def blur_point(image: np.array, point: Tuple[int]) -> np.array:
    image[point[1]][point[0]] = WHITE

#if needed find average using a window around the point, then get average
#color of the window to blur the chicken point.
def find_average(image: np.array, point: Tuple[int]) -> np.array:
    pass


def main() -> None:
    directory = './RawChickenData/'
    images: str = sorted(os.listdir(directory))
    numpy_pic_array: List[np.array] = []
    
    try:
        os.mkdir('./ModifiedChickenData')
    except:
        shutil.rmtree('./ModifiedChickenData')
        os.mkdir('./ModifiedChickenData')
    
    for img in images:
        pic:Image = Image.open(directory+img)
        numpy_pic: np.array = np.asarray(pic)
        
        for p in find_points(img):
            blur_point(numpy_pic, p)
    
        numpy_pic_array.append(numpy_pic)

    for index, modified_img in enumerate(numpy_pic_array):
        save_img = Image.fromarray(modified_img)
        save_img.save(f'./ModifiedChickenData/{index}.png')



 
if __name__ == '__main__':
    main()