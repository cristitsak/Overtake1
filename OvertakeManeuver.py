import configparser
import cv2
import numpy as np
import math
from detect import VehicleDetection

class OvertakeManeuver:
    def _init_(self, logger) -> None:
        self.log = logger
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.vehicle_detection = VehicleDetection(logger)
        self.refrence_point = None

    def OvertakeDesicion(self, frame):
        """Funcion to decide if the vehicle should overtake or not
        takes into account obstacles from back, side and front
        """
        pass

    def SteerCalculation(self, frame):
        _, _, vd_info = self.vehicle_detection.detect(frame)
        
        if vd_info("flag"):
            x , y, w, h = vd_info["bbox"]
            observed_point = np.array([x + w/2, y + h/2]) # the y values increase or decrease?

            if self.refrence_point is None:
                self.refrence_point = np.array(0.95 * frame.shape[1], observed_point[1])

            dist = math.dist(observed_point, self.refrence_point)
            if dist < frame.shape[1] * 0.1 :
                #follow line detection algorithm
                pass
            else:
                #set steering angle
                #also save maxinum steering angle
                pass