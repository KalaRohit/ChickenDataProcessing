import os
import shutil
import sys
import numpy as np
import cv2
import copy
from lama_cleaner.model.lama import LaMa
from typing import List, Tuple

WHITE = [255, 255, 255]
NO_OF_COLORS = 20
COLORS = {}

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
    res = cv2.bitwise_not(image_cpy, image_cpy, mask=mask)

    # cv2.imshow("testing", res)
    # cv2.waitKey(0)

    coords = cv2.findNonZero(mask)  
    coords = np.array(coords)
    coords = coords.squeeze(1)
    
    return coords

#Function blur_points:
#Blurs the point passed in on the image provided.
#Params: image: BGR matrix representation of image, List of two elements x,y 
#        which are the coordinates on the image to be blurred.
#Output: None, does it inplace on the image provided. 
def blur_point(image: np.array, blurred_image, point: List[int]) -> None:
    cv2.imwrite('test.png', blurred_image)
    image[point[1]][point[0]] = blurred_image[point[1]][point[0]]


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
        blurred_image = cv2.blur(numpy_pic, (101,101), 1)
        for p in find_points(numpy_pic):
            blur_point(numpy_pic, blurred_image, p)
    
        numpy_pic_array.append(numpy_pic)

    for index, modified_img in enumerate(numpy_pic_array):
        print(f'writing: {index}.png')
        cv2.imwrite(f'./ModifiedChickenData/{index}.png', modified_img)

    
 
if __name__ == '__main__':
    main()