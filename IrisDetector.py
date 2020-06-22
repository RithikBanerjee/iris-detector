
import os
import cv2
import numpy
from math import hypot


eye_images = []
image_names = []

#running the complete set of image database from the folder "DemoData"
for filename in os.listdir("DemoData"):
    if filename is not None:
        image = cv2.imread(os.path.join("DemoData", filename), 1)
        eye_images.append(image)
        image_names.append(filename.split('.')[0])

for i in range(len(eye_images)):
    current_image_name = image_names[i]
    current_image = eye_images[i]
    #show input image
    display_image(f"input_{current_image_name}", current_image)

    #first step grabcut
    background_removed_image = grab_cut(current_image)

    #second step noise reduction for circular hough transform
    #noise reduction on grayscaled image
    grayscaled_image = grayscale(background_removed_image)
    display_image(f"grab_cut_${current_image_name}", grayscaled_image)
    inverted_grayscaled_image = invert_grayscale(grayscaled_image)
    
    #enhance black pixels intensity
    blackhat_image = balckhat(inverted_grayscaled_image)
    display_image(f"black_hat_{current_image_name}", blackhat_image)

    #remove surface reflection points
    removed_refection = cv2.add(inverted_grayscaled_image, blackhat_image)
    image_without_reflection = median_blur(removed_refection)
    display_image(f"median_blur_{current_image_name}", image_without_reflection)

    #third step is to enhance the edges of the image
    edged_image = canny_edge(image_without_reflection)
    display_image(f"CED_{current_image_name}", edged_image)

    #fourth step get circular images
    circles = hough_circle(edged_image)
    if circles is not None:
        #fifth step mark circles co-ordinates
        inner_circle = draw_circle(circles, current_image)
        display_image(f"hough_circle_{current_image_name}", current_image)
        x, y = current_image.shape

        #crop iris part from grayscale image
        crop_image(x, y, inner_circle, grayscaled_image)
        display_image(f"output_{current_image_name}", grayscaled_image)
    cv2.destroyAllWindows()


#show image window
def display_image(caption, image):
    cv2.imshow(caption, image)
    cv2.waitKey(0)


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
    removed_background_image = rawImage * (normalization_mask[:,:,numpy.newaxis])
    return removed_background_image


#grayscale of image
def grayscale(raw_image):
    mode_of_operation = cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(raw_image, mode_of_operation)


#pixel averaging of grayscale images
def invert_grayscale(grayscaled_image):
    return cv2.bitwise_not(grayscaled_image)


#noise reduction technique 'balckhat'
def balckhat(grayscaled_image):
    structure_kernel = numpy.ones((5, 5), numpy.uint8)
    mode_of_operation = cv2.MORPH_BLACKHAT
    return cv2.morphologyEx(inverted_grayscaled_image, mode_of_operation, structure_kernel)


#pixel with high white intensity are averaged
def median_blur(raw_image):
    image_without_reflection = cv2.medianBlur(removed_refection, 5)
    cv2.equalizeHist(image_without_reflection)
    return image_without_reflection


#detect edges by pixel intensity
def canny_edge(raw_image):
    region_of_interest = cv2.bitwise_not(raw_image)
    mode_of_operation = cv2.THRESH_BINARY_INV
    retval, thresholded_image = cv2.threshold(region_of_interest, 50, 255, mode_of_operation)
    return cv2.Canny(thresholded_image, 200, 100)


#detects circular edges
def hough_circles(edged_image):
    mode_of_operation = cv2.HOUGH_GRADIENT
    return cv2.HoughCircles(edged_image, mode_of_operation, 1, 20, param1 = 200, param2 = 20, minRadius = 0)


#draws circles as per the co-ordinates given
def draw_circle(circles, raw_image):
    inner_circle = numpy.uint16(numpy.around(circles[0][0])).tolist()
    cv2.circle(raw_image, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 255, 0), 1)
    return inner_circle


#get correct circle and crop image
def crop_image(x_coordinate, y_coordinate, inner_circle, raw_image):
    for j in range(x_coordinate):
        for k in range(y_coordinate):
            if hypot(k - inner_circle[0], j - inner_circle[1]) >= inner_circle[2]:
                raw_image[j, k] = 0
