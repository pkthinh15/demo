import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils
import numpy as np
import matplotlib.pylab as plt
from imutils import contours
from scipy.fftpack import dct, idct


# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    


def s(img):
    plt.imshow(img)
    # plt.axis("off")
    plt.show()

img = cv2.imread("input/bang_diem.png")

import cv2

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def extract_digit(inputImage):
    """
    return list segment
    """
    lst_segment = []
    # Read Input image
    # inputImage = cv2.imread("digit.png")
    
    # Deep copy for results:
    inputImageCopy = inputImage.copy()
    
    # Convert BGR to grayscale:
    grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    
    # Threshold via Otsu:
    threshValue, binaryImage = cv2.threshold(grayscaleImage, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Flood-fill border, seed at (0,0) and use black (0) color:
    cv2.floodFill(binaryImage, None, (0, 0), 0)
    
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for the outer bounding boxes (no children):
    for _, c in enumerate(contours):
    
        # Get the bounding rectangle of the current contour:
        boundRect = cv2.boundingRect(c)
    
        # Get the bounding rectangle data:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]
    
        # Estimate the bounding rect area:
        rectArea = rectWidth * rectHeight
    
        # Set a min area threshold
        minArea = 10
        # Filter blobs by area:
        if rectHeight>5 and rectHeight<20 and rectWidth<30:
    
            # Draw bounding box:
            color = (0, 255, 0)
            cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                          (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)
            # Crop bounding box:
            if rectWidth>20:#extract 2 digits
                first_digit = inputImage[rectY:rectY+rectHeight,rectX:rectX+int(rectWidth/2)]
                second_digit = inputImage[rectY:rectY+rectHeight,rectX+int(rectWidth/2):rectX+rectWidth]
                s(first_digit)         
                s(second_digit)      
            else:
                currentCrop = inputImage[rectY:rectY+rectHeight,rectX:rectX+rectWidth]
                s(currentCrop)

def save_image(img,save_path="digit.png"):
    from PIL import Image
    im = Image.fromarray(img)
    im.save(save_path)
