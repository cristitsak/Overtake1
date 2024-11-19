import configparser
import os
import time

import cv2
import numpy as np
from jetson_inference import segNet
from jetson_utils import cudaAllocMapped, cudaDeviceSynchronize, cudaFromNumpy, cudaToNumpy


class SegmentationBuffers:
    def __init__(self, net):
        self.net = net
        self.mask = None
        self.use_mask = True

    @property
    def output(self):
        return self.mask

    def alloc(self, shape, format):
        self.mask = cudaAllocMapped(width=shape[1], height=shape[0], format=format)


class VehicleDetection:
    def __init__(self, logger) -> None:
        self.log = logger
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        # load the segmentation network
        network_dir = self.config["VEHICLE_DETECT"].get("model_path")
        self.net = segNet(
            model=os.path.join(network_dir, "fcn_resnet18.onnx"),
            labels=os.path.join(network_dir, "classes.txt"),
            colors=os.path.join(network_dir, "colors.txt"),
            input_blob="input_0",
            output_blob="output_0",
        )

        self.args = {
            "filter_mode": "linear",
            "ignore_class": "void",
            "alpha": 255.0,
            "mask_lower_bound": np.array(
                [int(value) for value in self.config["VEHICLE_DETECT"].get("lower_bound").split(", ")]
            ),
            "mask_upper_bound": np.array(
                [int(value) for value in self.config["VEHICLE_DETECT"].get("upper_bound").split(", ")]
            ),
            "mask_rectangles": self.config["VEHICLE_DETECT"].getboolean("mask_rect"),
            "keep_track": self.config["VEHICLE_DETECT"].getboolean("keep_track"),
            "overlap_thres": self.config["VEHICLE_DETECT"].getfloat("overlap_thres"),
        }

        # set the alpha blending value
        self.net.SetOverlayAlpha(self.args["alpha"])

        # create buffer manager
        self.buffers = SegmentationBuffers(self.net)

        width = self.config["CAM"].getint("imgW")
        height = self.config["CAM"].getint("imgH")
        w_top = self.config["VEHICLE_DETECT"].getfloat("w_top") * width
        h_top = self.config["VEHICLE_DETECT"].getfloat("h_top") * height
        w_base = self.config["VEHICLE_DETECT"].getfloat("w_base") * width
        h_base = self.config["VEHICLE_DETECT"].getfloat("h_base") * height
        self.mask_polygon = np.array(
            [
                [
                    (w_top, h_top),
                    (w_base, h_base),
                    (width - w_base, h_base),
                    (width - w_top, h_top),
                ]
            ]
        )

        self.bounding_margin = self.config["VEHICLE_DETECT"].getint("bounding_margin")
        self.contour_area_threshold = self.config["VEHICLE_DETECT"].getint("contour_area_threshold")

    def compute_overlap_ratio(self, rect1, rect2):
        """
        Computes the overlap ratio between two bounding rectangles.
        """
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
        y2 = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = rect1[2] * rect1[3] + rect2[2] * rect2[3] - intersection_area
        overlap_ratio = intersection_area / union_area

        return overlap_ratio

    def detect(self, color_img, depth_img, cam="S"):

        mask = np.zeros_like(color_img)
        detected = False
        veh_detected = {
            "flag": detected,
            "bbox": None,
            #"distance": float("inf"),
            "name": "Vehicle",
        }

        cv2.fillPoly(mask, np.int32([self.mask_polygon]), (255, 255, 255))
        color_masked = cv2.bitwise_and(color_img, mask)

        img_input = cudaFromNumpy(color_masked)

        # allocate buffers for this size image
        self.buffers.alloc(img_input.shape, img_input.format)

        # process the segmentation network
        self.net.Process(img_input, ignore_class=self.args["ignore_class"])

        # generate the mask
        if self.buffers.mask:
            self.net.Mask(self.buffers.mask, filter_mode=self.args["filter_mode"])

        # render the output image
        numpy_output = cudaToNumpy(self.buffers.output)
        rgb_output = cv2.cvtColor(numpy_output, cv2.COLOR_BGR2RGB)

        segmentation_mask = cv2.inRange(
            rgb_output, self.args["mask_lower_bound"], self.args["mask_upper_bound"]
        )

        if self.args["mask_rectangles"]:
            contours, hierarchy = cv2.findContours(
                segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            bounding_rects = []

            contour_areas = [cv2.contourArea(contour) for contour in contours]

            max_area = max(contour_areas) if contour_areas else -float("inf")

            if max_area > self.contour_area_threshold:
                contour = contours[contour_areas.index(max_area)]

                x, y, w, h = cv2.boundingRect(contour)
                bbox = {"x": x, "y": y, "w": w, "h": h}
                bounding_rects.append(
                    (
                        x - self.bounding_margin,
                        y - self.bounding_margin,
                        w + self.bounding_margin,
                        h + self.bounding_margin,
                    )
                )

                detected = True

            
                if np.any(depth_img):
                    distance = self.help.find_distance(depth_img, bbox, cam)
                else:
                    distance = float("inf")
                

                veh_detected = {
                    "flag": detected,
                    "bbox": bbox,
                    "distance": distance,
                    "name": "Vehicle",
                    "timestamp": time.time(),
                }

            mask = np.zeros_like(color_img[:, :, 0])

            if self.args["keep_track"]:
                prev_rects = []

                for rect in bounding_rects:

                    is_match = False
                    for prev_rect in prev_rects:
                        overlap_ratio = self.compute_overlap_ratio(rect, prev_rect)
                        if overlap_ratio > self.args["overlap_thres"]:
                            is_match = True
                            prev_rect[:] = rect
                            break

                    if not is_match:
                        prev_rects.append(rect)

                bounding_rects = prev_rects

            for rect in bounding_rects:
                x, y, w, h = rect
                mask[y : y + h, x : x + w] = 255

        else:
            mask = segmentation_mask

        masked_frame = cv2.bitwise_and(color_img, color_img, mask=mask)

        # self.log.debug(f"FPS >>> {self.net.GetNetworkFPS()}")
        cudaDeviceSynchronize()

        return masked_frame, bounding_rects, veh_detected

    def visualize_results(self, img_out, detection):

        if detection["flag"]:
            bbox = detection["bbox"]

            cv2.rectangle(
                img_out,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
                color=(255, 255, 0),
                thickness=10,
            )
            cv2.putText(
                img=img_out,
                text="Vehicle",
                org=(bbox["x"], bbox["y"] - 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=2,
                color=(255, 255, 0),
            )
