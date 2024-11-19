import os
from os import listdir
import cv2
import numpy as np
#from detect import VehicleDetection


"""
def find_white_dot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=300, param1=70, param2=40, minRadius=10, maxRadius=60)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            return (x, y)
    return None

for image in images:
    img = cv2.imread(f"frames/{folder}/{image}")
    if img is None:
        continue
    dot_position = find_white_dot(img)
    if dot_position:
        cv2.circle(img, dot_position, 5, (0, 0, 255), -1)
    cv2.imshow("Image with Dot", img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
"""

folder = "WhiteCar"
images = os.listdir("frames/" + folder)
images = sorted(images)

for image in images:
    frame = cv2.imread(f"frames/{folder}/{image}")
    cv2.imshow("Frame", frame)
    cv2.waitKey(100)

"""
    V_Detection = VehicleDetection(frame)
    mask, rect, detection_dict = VehicleDetection.detect(V_Detection, frame)
    if detection_dict["flag"]:
        print("Vehicle Detected")
    VehicleDetection.visualize_results(V_Detection, frame, detection_dict)
    """