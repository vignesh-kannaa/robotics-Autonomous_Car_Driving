from rasrobot import RASRobot
import numpy as np
import time
import math
import cv2
import random

# The autonomous car system utilizes various technologies to achieve its functionality. 
# The system has an on-board RGB camera that captures images using red, green, and blue filters, enabling accurate color representation and object distinction based on color. 
# Through the camera, the yellow line on the road is detected, and the steering angle of the car is manipulated to follow it. 
# The Ackerman drive system, which controls the front wheels independently, allows for smooth and precise turning, and speed and steering angle adjustments are made based on the yellow line's position.

# the autonomous car system also employs a deliberation process to determine the appropriate action to take at intersections. 
# This process involves checking for the nearest node of intersection while making random turns. 
# By considering the available options and assessing the optimal path, the system can determine the most suitable course of action at any given intersection.

# In addition to the aforementioned technologies, the autonomous car's localization system leverages sensors like GPS to determine its current location and aid in task planning, such as moving to the charging area. 
# The path and motion planning module takes into account the configuration of intersection points, which can be dynamically updated based on changes to the map. 
# Utilizing the compass direction between two points, such as the car's current location and the nearest intersection, the system directs the car towards the charging area, enabling efficient and accurate navigation.

class MyRobot(RASRobot):
    def __init__(self): 
        super(MyRobot, self).__init__()
        self.turn_time = 0
        # speed limit
        self.NORMAL_SPEED = 40
        self.TURNING_SPEED = 25
        # low battery percent -> 40%
        self.LOW_BATTERY = False
        self.lowBattery = 0.4 * (self.time_to_live)        
        print('intiate low battery level: ',self.lowBattery)
        # steering angle for turnings
        self.prev_steering_angle = 0
        self.car_turning_angle = 0
        self.RIGHT_TURN = 0.25
        self.LEFT_TURN = -0.25
        
        # the below configurations can be updated for new maps
        # GPS coordinates of target location
        self.TARGET_LAT = -215
        self.TARGET_LON = -145  
        # defining junction as node
        self.NodeA = (-44,33)
        self.NodeB = (43,-54)
        self.NodeC = (105,79)
        self.NodeD = (-215,-145)
        self.currentNode = None
        self.targetNode = None
        self.NodePath = {
            self.NodeA: self.NodeC,
            self.NodeB: self.NodeC,
            self.NodeC: self.NodeD,
        }
        # Initialise and resize a new window
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 256 * 2, 128 * 2)
        

    def run(self):

        while self.tick():
            print(f"{'low fuel' if self.LOW_BATTERY else 'Time to live'}: {round(self.time_to_live, 2)}")
            # checking low battery
            self.LOW_BATTERY = True if self.time_to_live <= self.lowBattery else False
            # setting up the time for the car to turn
            self.update_turn_time()
            # controlling speed based on car turn
            speed = self.NORMAL_SPEED if not self.turn_time else self.TURNING_SPEED
            # Processing image based on yellow line in the road
            self.yellowLineEnded = False
            edges = self.process_image()
            steering_angle = self.follow_yellow_lane(edges)
            steering_angle = self.check_car_turn(steering_angle)
            # setting up the speed and steering angle
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)
            # Display the output
            output = np.dstack((edges, edges, edges))
            cv2.imshow('output', output)
            cv2.waitKey(1)

    def update_turn_time(self,):
        if self.turn_time > 0:
            self.turn_time -= 1

    def process_image(self,):
        # Get the camera image and convert it to grayscale
        image = self.get_camera_image()
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Create mask to extract yellow pixels
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 130, 200])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Apply Gaussian blur to mask
        mask_blur = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
        # Apply Canny edge detection to mask
        edges = cv2.Canny(mask_blur, 100, 200)
        return edges

    def follow_yellow_lane(self, edges):
        # restricting the view(by croping the image to only look at the bottom quarter) since zebra cross is in yello
        height, width = edges.shape
        cropped = edges[3 * height // 4 : height, :]
        # Get the indices of the white pixels
        indices = np.where(cropped == 255)
        # Check if there are any white pixels in the image
        if len(indices[0]) == 0:
            return 0
        # Compute the center of the white pixels
        center = np.mean(indices[1])
        # Compute the deviation from the center of the image
        deviation = center - width / 2
        # Compute the steering angle
        steering_angle = deviation / (width / 2)
        return steering_angle

   
    def check_car_turn(self, steering_angle):
        # check if yellow line is ended
        if steering_angle == 0:
            self.yellowLineEnded = True
            steering_angle = self.car_turning_angle
        steering_angle_diff = abs(steering_angle - self.prev_steering_angle)
        # not to update the angle for period of time (till turning completes)
        if (
            self.yellowLineEnded
            and not bool(steering_angle_diff)
            and not self.turn_time
        ):
            self.check_latest_junction()
            if  not self.LOW_BATTERY:
            # random turn
                random_no = random.randint(0, 1)
                self.car_turning_angle = self.LEFT_TURN if random_no == 0 else self.RIGHT_TURN
            elif self.currentNode:
            # go to the nearest node of charging node
                self.targetNode = self.NodePath[self.currentNode]
                self.car_turning_angle = self.chargeDirection(self.currentNode, self.targetNode)
            else:
                self.car_turning_angle = self.RIGHT_TURN
            self.turn_time = 100
            
        self.prev_steering_angle = steering_angle
        return steering_angle
        
    def check_latest_junction(self,):
        # check the latest junction point car visited
        gps = self.get_gps_values()
        lat, lon, _ = gps
        # to get the location with threshold of 5
        # A node range
        if -48 <= lat <= -40 and 30 <= lon <= 36:
            self.currentNode = self.NodeA
        # B node range
        if 40 <= lat <= 50 and -57 <= lon <= -50:
            self.currentNode = self.NodeB
        # C node range    
        if 100 <= lat <= 110 and 75 <= lon <= 85:
            self.currentNode = self.NodeC
        
     
    def chargeDirection(self, source, destination):
            # using current location & target location to find turning angle
            # GPS coordinates of current location
            current_lat, current_lon = source
            target_lat, target_lon = destination
            # Calculate vector between current location and target location
            delta_lat = target_lat - current_lat
            delta_lon = target_lat - current_lon
            # Calculate direction from current location to target location            
            degrees_temp = (math.atan2(delta_lat, delta_lon)/math.pi*180 )
            # Determine which direction to turn
            return self.RIGHT_TURN if degrees_temp < 0 else self.LEFT_TURN


robot = MyRobot()
robot.run()
