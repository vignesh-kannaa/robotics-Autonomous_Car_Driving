from rasrobot import RASRobot
import numpy as np
import time
import math
import cv2
import yolov5


# The autonomous car system utilizes various technologies to achieve its functionality. 
# The system has an on-board RGB camera that captures images using red, green, and blue filters, enabling accurate color representation and object distinction based on color. 
# Through the camera, the yellow line on the road is detected, and the steering angle of the car is manipulated to follow it. 
# The Ackerman drive system, which controls the front wheels independently, allows for smooth and precise turning, and speed and steering angle adjustments are made based on the yellow line's position.

# The autonomous car system relies on perception, which involves interpreting sensory information from the environment to understand and interact with it. 
# The car's perception module uses YOLO technology to detect objects, including stop signs, on the road. As part of the behavior coordination programmed into the system, when the car's perception module detects a stop sign, the car comes to a complete stop for 1 second.

# the autonomous car system also employs a deliberation process to determine the appropriate action to take at intersections. 
# This process involves checking for the nearest node of intersection while making random turns. 
# By considering the available options and assessing the optimal path, the system can determine the most suitable course of action at any given intersection.

# In addition to the aforementioned technologies, the autonomous car's localization system leverages sensors like GPS to determine its current location and aid in task planning, such as moving to the charging area. 
# The path and motion planning module takes into account the configuration of intersection points, which can be dynamically updated based on changes to the map. 
# Utilizing the compass direction between two points, such as the car's current location and the nearest intersection, the system directs the car towards the charging area, enabling efficient and accurate navigation.


# Detect stop sign using YOLOv5
# the small version of yolo is used, thus accuracy may not be so good. 
# To achieve better accuracy, other versions of YOLO could be used, but this may come at the cost of slower processing speed.
model = yolov5.load('yolov5s.pt')
import random

class MyRobot(RASRobot):
    def __init__(self): 
        super(MyRobot, self).__init__()
        self.sleeptime = 0
        # speed limit
        self.NORMAL_SPEED = 40
        self.TURNING_SPEED = 25
        # low battery percent -> 40%
        self.LOW_BATTERY = False
        self.lowBattery = 0.4 * (self.time_to_live)        
        print('Low battery intiate: ',self.lowBattery)
        # steering angle for turnings
        self.prev_steering_angle = 0
        self.car_turning_angle = 0
        self.RIGHT_TURN = 0.25
        self.LEFT_TURN = -0.25
        # stop sign initiate
        self.stop_sign_detected = False
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
        cv2.resizeWindow("output", 256 * 2, 128 * 2)
        

    def run(self):
        counter = 1
        while self.tick():
            counter +=1
            self.LOW_BATTERY = True if self.time_to_live <= self.lowBattery else False
            print(f"{'low fuel' if self.LOW_BATTERY else 'Time to live'}: {round(self.time_to_live, 2)}")
            # setting up the time to sleep any function
            self.update_sleep_time()
            # controlling speed based on car turn
            speed = self.NORMAL_SPEED if not self.sleeptime else self.TURNING_SPEED
            # Processing image based on yellow line in the road
            self.yellowLineEnded = False
            # Get the camera image and convert it to grayscale
            image = self.get_camera_image()
            edges = self.process_image(image)
            steering_angle = self.follow_yellow_lane(edges)
            steering_angle = self.check_car_turn(steering_angle)
            # setting up the speed and steering angle
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)
            # print(f'Time to live: {round(self.time_to_live,2)}')
            self.check_stop_sign(counter, image)
            
            # Display the output
            output = np.dstack((edges, edges, edges))
            cv2.imshow('output', output)
            cv2.waitKey(1)

    def update_sleep_time(self,):
        if self.sleeptime > 0:
            self.sleeptime -= 1

    def process_image(self,image):
        # Get the camera image and convert it to grayscale
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

    def check_stop_sign(self, counter, image):
        # using counter to reduce the number of images model processes
        if counter % 15 == 0:
            if not self.stop_sign_detected:
                results = model(image, size=640)
            for result in results.xyxy[0]:
              # Check if the detected object is a stop sign (yolov5 has the class label of stop sign in class 11)
                if result[5] == 11:
                    self.stop_sign_detected = True
                    print('-Stop sign detected!-')
                    break

        if self.stop_sign_detected:
            # Stop the car for 1 second if sign is detected
            print('-Stopping car for 1 sec-')
            self.set_speed(0)
            self.set_steering_angle(0)
            time.sleep(1)
            self.stop_sign_detected = False
                
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
            and not self.sleeptime
        ):
            if not self.LOW_BATTERY:
            # random turn
                random_no = random.randint(0, 1)
                self.car_turning_angle = self.LEFT_TURN if random_no == 0 else self.RIGHT_TURN
            elif self.currentNode:
            # go to the nearest node of charging node
                self.targetNode = self.NodePath[self.currentNode]
                self.car_turning_angle = self.chargeDirection(self.currentNode, self.targetNode)
            else:
                self.car_turning_angle = self.RIGHT_TURN
            self.sleeptime = 100
            
        self.prev_steering_angle = steering_angle
        return steering_angle
        
    def checkLatestJunction(self,):
        gps = self.get_gps_values()
        lat, lon, _ = gps
        # A node range
        if -48 <= lat <= -40 and 30 <= lon <= 36:
            self.currentNode = self.NodeA
        # B node range
        if 40 <= lat <= 50 and -57 <= lon <= -50:
            self.currentNode = self.NodeB
        # C node range    
        if 100 <= lat <= 110 and 75 <= lon <= 85:
            self.currentNode = self.NodeC
            

    # find the compass direction of the charging area
    # Reference: https://www.analytics-link.com/post/2018/08/21/calculating-the-compass-direction-between-two-points-in-python
    # using the above reference, I have modified the code below for my use case
    def chargeDirection(self, source, destination):
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
