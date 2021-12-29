import cv2
import numpy as np

###################################################################
# # Just for debugging the webcam - get video and display
# cap = cv2.VideoCapture(0) #Capture video source zero `/dev/video0`
# cap.set(3, 1920)
# cap.set(4, 1080)
#
# while True:
#     success, vid = cap.read()
#     cv2.imshow('Video', vid)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
######################################################################

#############################
webcamWidth = 1920
webcamHeight = 1080
#############################

# Capture video from your webcam
cap = cv2.VideoCapture(0) # Capture video source zero `/dev/video0`
cap.set(3, webcamWidth)
cap.set(4, webcamHeight)
cap.set(10, 130) # Set brightness


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgEdges = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5)) # Add contour thickness to make it more visible
    imgDilated = cv2.dilate(imgEdges, kernel, iterations=2)
    imgThresh= cv2.erode(imgDilated, kernel, iterations=1)

    return imgThresh


def getContours(img): # Retrieve contours from detected shapes
    # biggest = np.array([])
    biggest = np.zeros((4, 1, 2), np.int32)
    maxArea = 0
    contours, hierachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt) # Get areas for all contours
        if area > 5000: # Set threshold to exclude noise
            peri = cv2.arcLength(cnt, True) # Get contour perimeter
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True) # Approximate polygonal curve
            if area > maxArea and len(approx) == 4: # Loop until biggest contour is found
                biggest = approx
                maxArea = area

    cv2.drawContours(imgBackground, biggest, -1, (0, 255, 255), 20) # Print corner points of biggest contour

    return biggest


def reorder (cornerPoints):
    # getWarp() expects the points to be left-top, r-t, l-b, r-b
    # To get this order sum up their x/y position and order ascending
    cornerPoints = cornerPoints.reshape((4, 2))
    cornerPointsNew = np.zeros((4, 1, 2), np.int32)
    add = cornerPoints.sum(1)

    cornerPointsNew[0] = cornerPoints[np.argmin(add)]
    cornerPointsNew[3] = cornerPoints[np.argmax(add)]

    diff = np.diff(cornerPoints, axis=1)
    cornerPointsNew[1] = cornerPoints[np.argmin(diff)]
    cornerPointsNew[2] = cornerPoints[np.argmax(diff)]

    return cornerPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest) # reorder() takes care that the points are in the correct order
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    # Crop unclean edges -20px and stretch back to original size
    imgCropped = imgOutput[20:imgOutput.shape[0]-20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgOutput


while True:
    success, img = cap.read()
    imgBackground = img.copy()

    imgScan = preProcessing(img) # Grayscale + find edges
    biggest = getContours(imgScan) # Find the biggest shape and select corner points
    # print(biggest) # Print found corner points
    scanRectified = getWarp(img, biggest) # Remove perspective distortion

    cv2.imshow('Contours', imgScan)  # Show found contours
    cv2.imshow('Scan Corners', imgBackground) # Show found corner points
    cv2.imshow('Document', scanRectified) # Show de-warped, cropped image

    if cv2.waitKey(1) & 0xFF == ord('q'): # Keep running until you press `q`
        break
