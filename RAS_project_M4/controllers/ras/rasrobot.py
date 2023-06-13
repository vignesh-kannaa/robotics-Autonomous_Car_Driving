#
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
# PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
#
import numpy as np

from controller import Robot
from vehicle import Driver

import cv2

class RASRobot(object):
    """
    This is the class you will use to interact with the car.
    PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    PLEASE DO NOT MODIFY THE CODE. WHEN MARKING, THIS FILE WILL BE OVERWRITTEN.
    """
    def __init__(self):
        self.__robot = Driver()
        self.__timestep = int(self.__robot.getBasicTimeStep())

        self.__camera = self.__robot.getDevice("camera")
        self.__camera.enable(self.__timestep)
        
        self.__gps = self.__robot.getDevice("gps")
        self.__gps.enable(self.__timestep)
        
        self.__time_to_live = 240.
       
    @property
    def time_to_live(self):
        return self.__time_to_live

    def get_camera_image(self):
        """
        This method returns a NumPy array representing the latest image captured by the car's camera.
        It will have 64 rows, 128 columns and 4 channels (red, green, blue, alpha).
        """
        return np.frombuffer(self.__camera.getImage(), np.uint8).reshape((128,256,4))

    def set_steering_angle(self, angle):
        """
        This is just a proxy for the Webot's API call. It sets the steering angle of the car.
        For more information: https://cyberbotics.com/doc/automobile/driver-library?tab-language=python#wbu_driver_set_steering_angle
        """
        self.__robot.setSteeringAngle(angle)

    def set_speed(self, speed):
        """
        This is just a proxy for the Webot's API call. It sets the speed of the car.
        For more information: https://cyberbotics.com/doc/automobile/driver-library?tab-language=python#wbu_driver_set_cruising_speed
        """
        self.__robot.setCruisingSpeed(speed)
    
    def get_gps_values(self):
        """
        This is just a proxy for the Webot's API call. It gets the values of the GPS device.
        For more information: https://cyberbotics.com/doc/reference/gps?tab-language=python#wb_gps_get_values
        """
        return self.__gps.getValues()
    
    def tick(self):
        """
        You will call this method rather than the typical `step` method used by regular Webots controllers.
        """
        gps_values = self.get_gps_values()
        if -230 < gps_values[0] < -194 and  -155 < gps_values[1] < 220:
            self.__time_to_live += 0.001 * self.__robot.getBasicTimeStep() * 10
            if self.__time_to_live > 240.:
                self.__time_to_live = 240.
        else:
            self.__time_to_live -= 0.001 * self.__robot.getBasicTimeStep()
            if self.__time_to_live <= 0:
                self.__time_to_live = 0
                self.set_speed(0)
                


        if self.__robot.step() == -1:
            return False
            

        return True
        
