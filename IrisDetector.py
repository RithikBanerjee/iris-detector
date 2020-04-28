import cv2 
import numpy
import os
import math
from math import hypot
gamma = -48
#gamma is -48 for UBIRIS database


#method to differentiat foreground and background
def grab_cut(raw_image):
    number_of_interation = 5
    channel_mask = numpy.zeros((raw_image.shape[0], raw_image.shape[1]), numpy.uint8)
    region_of_interest = (50, 50, 450, 290)
    background_model = numpy.zeros((1,65),np.float64)
    foreground_model = numpy.zeros((1,65),np.float64)
    mode_of_operation = cv2.GC_INIT_WITH_RECT
    cv2.grabCut(raw_image, channel_mask, region_of_interest, background_model, foreground_model, number_of_interation, mode_of_operation)
    normalization_mask = np.where((channelMask == 2)|(channelMask == 0), 0, 1).astype('uint8')
    removed_background_image = rawImage*normalization_mask[:,:,numpy.newaxis]
    return removed_background_image


#noise canceling method (pre CHT processing) 
def noise_reduction(grayscaled_image, name_of_image):                             
    inverted_grayscaled_image = cv2.bitwise_not(grayscaled_image)    
    structure_kernel = numpy.ones((5, 5), numpy.uint8)
    mode_of_operation = cv2.MORPH_BLACKHAT
    blackhat_image = cv2.morphologyEx(inverted_grayscaled_image, mode_of_operation, structure_kernel)
    cv2.imshow("black_hat_"+name_of_image, blackhat_image)
    cv2.waitKey(0)
    
    removed_refection = cv2.add(inverted_grayscaled_image, blackhat_image)
    image_without_reflection = cv2.medianBlur(removed_refection, 5)
    cv2.equalizeHist(image_without_reflection)
    cv2.imshow("median_blur_"+name_of_image, image_without_reflection)
    cv2.waitKey(0)

    region_of_interest = cv2.bitwise_not(image_without_reflection)
    mode_of_operation = cv2.THRESH_BINARY_INV
    retval, thresholded_image = cv2.threshold(region_of_interest, 50, 255, mode_of_operation)
    edged_image = cv2.Canny(thresholded_image, 200, 100)
    cv2.imshow("CED_"+current_image_name, edged_image)
    cv2.waitKey(0)
    
    return edged_image

eye_images = []
image_names = []
 #running the complete set of image database from the folder "DemoData"
for filename in os.listdir("Demo Data"):
    if filename is not None:
        image = cv2.imread(os.path.join("Demo Data",filename), 1)
        eye_images.append(image)
        image_names.append(filename.split('.')[0])

for i in range(len(eye_images)):
    current_image_name = image_names[i]
    current_image = eye_images[i]
    cv2.imshow("input_"+current_image_name, current_image)
    cv2.waitKey(0)
    
    x, y, z = current_image.shape
    background_removed_image = Grabcut(current_image)
    mode_of_operation = cv2.COLOR_BGR2GRAY
    grayscaled_image = cv2.cvtColor(background_removed_image, mode_of_operation)
    cv2.imshow("grab_cut_"+current_image_name, grayscaled_image)
    cv2.waitKey(0)

    noise_removed_image = noise_reduction(grayscaled_image, current_image_name)
    mode_of_operation = cv2.HOUGH_GRADIENT
    circles = cv2.HoughCircles(noise_removed_image, mode_of_operation, 1, 20, param1 = 200, param2 = 20,minRadius =0)
   
    if circles is not None:
        inner_circle = numpy.uint16(numpy.around(circles[0][0])).tolist()
    cv2.circle(current_image ,(inner_circle[0],inner_circle[1]), inner_circle[2], (0,255,0), 1)
    cv2.imshow("hough_circle_"+current_image_name, current_image)
    cv2.waitKey(0)
    
#crop image by comparing each pixel dsitance from center with radius
    for j in range(x):   
        for k in range(y):
            if hypot(k-inner_circle[0], j-inner_circle[1]) >= inner_circle[2]: 
                grayscaled_image[j,k] = 0
    cv2.imshow("output_"+current_image_name, grayscaled_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

