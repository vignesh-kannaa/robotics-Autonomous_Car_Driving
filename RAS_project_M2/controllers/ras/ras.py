from rasrobot import RASRobot

import numpy as np
import time
import cv2
import yolov5


# The autonomous car system utilizes various technologies to achieve its functionality. 
# The system has an on-board RGB camera that captures images using red, green, and blue filters, enabling accurate color representation and object distinction based on color. 
# Through the camera, the yellow line on the road is detected, and the steering angle of the car is manipulated to follow it. 
# The Ackerman drive system, which controls the front wheels independently, allows for smooth and precise turning, and speed and steering angle adjustments are made based on the yellow line's position.

# The autonomous car system relies on perception, which involves interpreting sensory information from the environment to understand and interact with it. 
# The car's perception module uses YOLO technology to detect objects, including stop signs, on the road. As part of the behavior coordination programmed into the system, when the car's perception module detects a stop sign, the car comes to a complete stop for 1 second.


# Detect stop sign using YOLOv5
# the small version of yolo is used, thus accuracy may not be so good. 
# To achieve better accuracy, other versions of YOLO could be used, but this may come at the cost of slower processing speed.
model = yolov5.load('yolov5s.pt')

class MyRobot(RASRobot):
    def __init__(self):
        super(MyRobot, self).__init__()
        # Initialise speed limit
        self.NORMAL_SPEED = 30
        self.TURNING_SPEED = 20
        # Initialise steering angle for turnings
        self.prev_steering_angle = 0
        self.car_turning_angle = 0
        self.RIGHT_TURN = 0.25
        # stop sign initiate
        self.stop_sign_detected = False
        # Initialise and resize a new window 
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 256*2, 128*2)

    def run(self):
        counter = 1
        while self.tick():
            counter +=1
            speed = self.NORMAL_SPEED 
            # Processing image based on yellow line in the road
            self.yellowLineEnded = False
            # Get the camera image and convert it to grayscale
            image = self.get_camera_image()
            edges = self.process_image(image)
            # get the steering angle based on the yellow line
            steering_angle = self.follow_yellow_lane(edges)
             # checking whether to turn the steering based on the previous steering angle and yellow line
            steering_angle = self.check_car_turn(steering_angle)
            
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)

            self.check_stop_sign(counter, image)

            # Display the output image with the detected edges
            output = np.dstack((edges, edges, edges))
            cv2.imshow('output', output)
            cv2.waitKey(1)
            
    def process_image(self,image):

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
        ):
            self.car_turning_angle = self.RIGHT_TURN
        self.prev_steering_angle = steering_angle
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
                    print('Stop sign detected!')
                    break

        if self.stop_sign_detected:
            # Stop the car for 1 second if sign is detected
            print('Stopping car for 1 sec')
            self.set_speed(0)
            self.set_steering_angle(0)
            time.sleep(1)
            self.stop_sign_detected = False    
            
robot = MyRobot()
robot.run()


