import os
import shutil
import sys
import numpy as np
import cv2
import copy

from typing import List, Tuple

WHITE = [255, 255, 255]
NO_OF_COLORS = 20
COLORS = {}

#read the csv file for the points. Return as (x,y) 
def find_points(image: np.array) -> List[Tuple[int]]:
    image_copy = copy.deepcopy(image)
    lower_blue = np.array([78,48,53])
    upper_blue = np.array([145,255,255])

    a = np.uint8([[[49, 53, 43]]])
    b = np.uint8([[[255, 0, 213]]])

    hsv1 = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    coord = cv2.findNonZero(mask)

    res = cv2.bitwise_not(image_copy, image_copy, mask=mask)


    
    coord = np.array(coord)
    coords = coord.squeeze(1)
    
    return coords


def blur_point(image: np.array, point: Tuple[int]) -> np.array:
    # image[point[1]][point[0]] = WHITE
    blurred_image = cv2.GaussianBlur(image, (101,101), 1)
    cv2.imwrite('test.png', blurred_image)
    image[point[1]][point[0]] = blurred_image[point[1]][point[0]]

    

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
        
    
    for img in images[:1]:
        pic = cv2.imread(directory+img, cv2.IMREAD_COLOR)
        numpy_pic: np.array = np.asarray(pic)
        
        for p in find_points(numpy_pic):
            blur_point(numpy_pic, p)
    
        numpy_pic_array.append(numpy_pic)

    f = find_points(numpy_pic_array[-1])


    # for index, modified_img in enumerate(numpy_pic_array):
    #     save_img = Image.fromarray(modified_img)
    #     save_img.save(f'./ModifiedChickenData/{index}.png')

    
 
if __name__ == '__main__':
    main()