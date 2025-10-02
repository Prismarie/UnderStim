import os
from math import gcd
import mss
import numpy as np
import cv2 as cv

resolutions_21_by_9 = (
    (2560, 1080),  # WFHD
    (3440, 1440),  # WQHD
    (3840, 1600),  # WQHD+
    (5120, 2160),  # WUHD
    (5760, 2400),  # 5K UW
    (7680, 3200),  # 7K UW
    (8640, 3600),  # 8K UW
    (10240, 4320), # 10K UW
)

# This is necessary to access files when it's compiled into an executable
def resource_path(relative_path):
    return os.path.join(os.path.abspath("."), relative_path)

def resolution_to_aspect_ratio_string(horizontal_resolution: int, vertical_resolution: int):
    if (horizontal_resolution, vertical_resolution) in resolutions_21_by_9:
        return "21:9"
    greatest_common_divisor = gcd(horizontal_resolution, vertical_resolution)
    return f"{horizontal_resolution // greatest_common_divisor}:{vertical_resolution // greatest_common_divisor}"

class ComputerVision:
    def __init__(self, coords, mask_names, print_detected_resolution=True):
        self.base_resolution = {"width": 1920, "height": 1080}
        self.base_aspect_ratio = self.base_resolution["width"] / self.base_resolution["height"]
        self.print_detected_resolution = print_detected_resolution

        # Create mss capture object
        self.sct = mss.mss()
        monitor = self.sct.monitors[1]  # default to primary monitor
        self.screenshot_region = monitor
        self.final_resolution = {"width": monitor["width"], "height": monitor["height"]}

        # Resolution / aspect ratio checks
        self.resolution_mismatch = self.final_resolution != self.base_resolution
        if self.print_detected_resolution:
            print(f"Detected monitor resolution as {self.final_resolution['width']}x{self.final_resolution['height']}.")

        self.final_aspect_ratio = self.final_resolution["width"] / self.final_resolution["height"]
        self.aspect_ratio_mismatch = self.final_aspect_ratio != self.base_aspect_ratio
        if self.aspect_ratio_mismatch:
            print(
                f"Detected monitor aspect ratio as "
                f"{resolution_to_aspect_ratio_string(self.final_resolution['width'], self.final_resolution['height'])}."
            )
            print(
                f"Please ensure your in-game aspect ratio is set to "
                f"{resolution_to_aspect_ratio_string(self.base_resolution['width'], self.base_resolution['height'])}."
            )

        # Load templates and masks
        self.coords = coords
        self.templates = {
            key: cv.cvtColor(cv.imread(resource_path(f"data/t_{key}.png")), cv.COLOR_RGB2GRAY) for key in coords
        }
        self.mask_names = mask_names
        self.masks = {
            key: cv.cvtColor(cv.imread(resource_path(f"data/m_{key}.png")), cv.COLOR_RGB2GRAY) for key in mask_names
        }
        self.frame = []

    def grab_frame(self, monitor=1):
        screenshot = self.sct.grab(self.sct.monitors[monitor])
        img = np.array(screenshot)
        # Convert BGRA → BGR for OpenCV
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


    def start_capturing(self, target_fps=60):
        # Compatibility shim for old code that expected dxcam_cpp
        self.capture_frame()

    def stop_capturing(self):
        # No-op for compatibility (dxcam_cpp had this)
        pass

    def capture_frame(self):
        screenshot = self.sct.grab(self.screenshot_region)
        img = np.array(screenshot)
        if self.resolution_mismatch:
            img = cv.resize(img, (self.base_resolution["width"], self.base_resolution["height"]))
        self.frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        self.resolution_mismatch = self.final_resolution != self.base_resolution
        if self.print_detected_resolution:
            print(f"Detected monitor resolution as {self.final_resolution['width']}x{self.final_resolution['height']}.")

        self.final_aspect_ratio = self.final_resolution["width"] / self.final_resolution["height"]
        self.aspect_ratio_mismatch = self.final_aspect_ratio != self.base_aspect_ratio
        if self.aspect_ratio_mismatch:
            print(f"Detected monitor aspect ratio as {resolution_to_aspect_ratio_string(self.final_resolution['width'], self.final_resolution['height'])}.")
            print(f"Please ensure your in-game aspect ratio is set to {resolution_to_aspect_ratio_string(self.base_resolution['width'], self.base_resolution['height'])}.")

    def crop(self, image, template_name, coords_override=None):
        if coords_override is None:
            return image[self.coords[template_name][0]:self.coords[template_name][1],
                         self.coords[template_name][2]:self.coords[template_name][3]]
        else:
            return image[coords_override[0]:coords_override[1], coords_override[2]:coords_override[3]]

    def match(self, template_name, coords_override=None):
        cropped_frame = self.crop(self.frame, template_name, coords_override)
        if template_name in self.mask_names:
            return cv.matchTemplate(cropped_frame, self.templates[template_name], cv.TM_CCOEFF_NORMED, mask=self.masks[template_name])
        else:
            return cv.matchTemplate(cropped_frame, self.templates[template_name], cv.TM_CCOEFF_NORMED)

    def detect_multiple(self, template_name, threshold=0.9):
        result = self.match(template_name)
        _, result = cv.threshold(result, threshold, 255, cv.THRESH_BINARY)
        return len(cv.findContours(result.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0])

    def detect_single(self, template_name, threshold=0.9, coords_override=None):
        result = self.match(template_name, coords_override)
        return np.nanmax(result) > threshold
