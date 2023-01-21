import os
import shutil
import sys
import numpy as np
import cv2
import copy
import torch

from typing import List, Tuple

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
    pass


def main() -> None:
    read_directory = './RawChickenData/'
    write_directory = './lama_images'

    images: str = sorted(os.listdir(read_directory))
    numpy_pic_array: List[np.array] = []
    
    try:
        os.mkdir('./lama_images')   #images and masks to be plugged into lama model
    except:
        shutil.rmtree('./lama_images')
        os.mkdir('./lama_images')
        
    for index, img in enumerate(images[:1]):
        pic = cv2.imread(read_directory+img, cv2.IMREAD_COLOR)
        numpy_pic: np.array = np.asarray(pic)
        mask = find_points(numpy_pic)
        print(mask.shape)
        cv2.imwrite(f'./lama_images/{index}.png', pic)
        cv2.imwrite(f'./lama_images/{index}_mask.png', mask)
        

    # for index, modified_img in enumerate(blurred_imgs):
    #     print(f'writing: {index}.png')
    #     cv2.imwrite(f'./ModifiedChickenData/{index}.png', modified_img)

    
 
if __name__ == '__main__':
    main()