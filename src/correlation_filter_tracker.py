import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.ex2_utils import get_patch
from src.ex3_utils import create_cosine_window, create_gauss_peak
from src.tracker import Tracker
#from src.ex2_utils import Tracker

class CorelationParams():
    def __init__(self,
                 alpha: float = 0.8,
                 sigma: float = 0.9,
                 lambd: float = 1,
                 enlarge_factor: float = 2):
        self.enlarge_factor = enlarge_factor
        self.alpha = alpha
        self.sigma = sigma
        self.lambd = lambd

class CorelationTracker(Tracker):
    parameters: CorelationParams

    def __init__(self, params: CorelationParams = CorelationParams()):
        self.parameters = params
        super().__init__()

    def name(self):
        return "Corelation_filter_tracker_sigma10"

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        region = [round(r) for r in region]

        if(region[2]%2 == 0):
            region[2] = region[2]-1
        if(region[3]%2 == 0):
            region[3] = region[3]-1

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor


        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (int(region[2]), int(region[3]))
        # Create cosine wave and gaussian peak only once
        self.cosine_wave = create_cosine_window(self.size)
        self.G = create_gauss_peak(self.size, self.parameters.sigma)
        self.FG = np.fft.fft2(self.G)

        # Corelation filter creation
        F, _ = get_patch(image, self.position, self.size)
        F  = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
        F = np.multiply(F, self.cosine_wave)
        self.H = self.create_filter(F)

    def track(self, image):
        F, mask = get_patch(image, self.position, self.size)
        F  = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
        F = F*mask
        F = np.multiply(F, self.cosine_wave)
        FF = np.fft.fft2(F)
        R = np.real(np.fft.ifft2(np.multiply(self.H, FF)))
        # Get max in array
        max_y, max_x = np.unravel_index(R.argmax(), R.shape)

        if(max_x > self.size[0]/2):
            max_x = max_x - self.size[0]
        if(max_y > self.size[1]/2):
            max_y = max_y - self.size[1]
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(R)
        old = self.position
        self.position = (self.position[0] + max_x, self.position[1] + max_y)
        # Update filter
        F, mask = get_patch(image, self.position, self.size)
        F  = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
        F = F*mask
        
        F = np.multiply(F, self.cosine_wave)
        now_H = self.create_filter(F)

        self.H = self.parameters.alpha*self.H + (1-self.parameters.alpha)*now_H

        return [self.position[0]-(self.size[0]/2), self.position[1]-(self.size[1]/2), self.size[0], self.size[1]]


    def create_filter(self, F: np.array) -> np.array:
        """A function that returns discriminative correlation filter based on the target patch.

        :param F: The target patch
        :type F: np.array
        :return: Discriminative correlation filter
        :rtype: np.array
        """
        
        FF = np.fft.fft2(F)
        conjugated_FF = np.conjugate(FF)
        H = np.divide(np.multiply(self.FG,conjugated_FF),
                           np.add(np.multiply(FF, conjugated_FF), self.parameters.lambd))

        return H


class CorelationTrackerLarger(Tracker):
    parameters: CorelationParams

    def __init__(self, params: CorelationParams = CorelationParams()):
        self.parameters = params
        super().__init__()

    def name(self):
        return "Corelation_filter_tracker_larger20"

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        # Round to integers
        region = [round(r) for r in region]

        # Make shure the sizes are odd
        if(region[2]%2 == 0):
            region[2] = region[2]-1
        if(region[3]%2 == 0):
            region[3] = region[3]-1

        #self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (int(region[2]), int(region[3]))
        self.search_size = (int(region[2]*self.parameters.enlarge_factor), int(region[3]*self.parameters.enlarge_factor))

        # Make sure the sizes are odd
        if(self.search_size[0]%2 == 0):
            self.search_size = (self.search_size[0]-1, self.search_size[1])
        if(self.search_size[1]%2 == 0):
            self.search_size = (self.search_size[0], self.search_size[1]-1)

        # Create cosine wave and gaussian peak only once
        self.cosine_wave = create_cosine_window(self.search_size)
        self.G = create_gauss_peak(self.search_size, self.parameters.sigma)
        self.FG = np.fft.fft2(self.G)

        # Corelation filter creation
        F, _ = get_patch(image, self.position, self.search_size)
        F  = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
        F = np.multiply(F, self.cosine_wave)
        self.H = self.create_filter(F)

    def track(self, image):
        F, mask = get_patch(image, self.position, self.search_size)
        F  = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
        F = F*mask
        F = np.multiply(F, self.cosine_wave)
        FF = np.fft.fft2(F)
        R = np.real(np.fft.ifft2(np.multiply(self.H, FF)))
        # Get max in array
        max_y, max_x = np.unravel_index(R.argmax(), R.shape)

        if(max_x > self.search_size[0]/2):
            max_x = max_x - self.search_size[0]
        if(max_y > self.search_size[1]/2):
            max_y = max_y - self.search_size[1]
        #min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(R)
        old = self.position
        self.position = (self.position[0] + max_x, self.position[1] + max_y)
        # Update filter
        F, mask = get_patch(image, self.position, self.search_size)
        F  = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
        F = F*mask
        
        F = np.multiply(F, self.cosine_wave)
        now_H = self.create_filter(F)

        self.H = self.parameters.alpha*self.H + (1-self.parameters.alpha)*now_H

        return [self.position[0]-(self.size[0]/2), self.position[1]-(self.size[1]/2), self.size[0], self.size[1]]


    def create_filter(self, F: np.array) -> np.array:
        """A function that returns discriminative correlation filter based on the target patch.

        :param F: The target patch
        :type F: np.array
        :return: Discriminative correlation filter
        :rtype: np.array
        """
        
        FF = np.fft.fft2(F)
        conjugated_FF = np.conjugate(FF)
        H = np.divide(np.multiply(self.FG,conjugated_FF),
                           np.add(np.multiply(FF, conjugated_FF), self.parameters.lambd))

        return H
