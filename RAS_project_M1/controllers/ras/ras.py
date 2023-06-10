from rasrobot import RASRobot
import numpy as np
import time
import cv2


# The autonomous car system utilizes various technologies to achieve its functionality. 
# The system has an on-board RGB camera that captures images using red, green, and blue filters, enabling accurate color representation and object distinction based on color. 
# Through the camera, the yellow line on the road is detected, and the steering angle of the car is manipulated to follow it. 
# The Ackerman drive system, which controls the front wheels independently, allows for smooth and precise turning, and speed and steering angle adjustments are made based on the yellow line's position.

class MyRobot(RASRobot):
    def __init__(self):
        super(MyRobot, self).__init__()
        self.turn_time = 0
        # Initialise speed limit
        self.NORMAL_SPEED = 40
        self.TURNING_SPEED = 30
        # Initialise steering angle for turnings
        self.prev_steering_angle = 0
        self.car_turning_angle = 0
        self.RIGHT_TURN = 0.25
        # Initialise and resize a new window 
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 256 * 2, 128 * 2)

    def run(self):
    
        while self.tick():
            # setting up the time to turn
            self.update_turn_time()
            # controlling speed based on car turn
            speed = self.NORMAL_SPEED if not self.turn_time else self.TURNING_SPEED
            # Processing image based on yellow line in the road
            self.yellowLineEnded = False
            edges = self.process_image()
            # get the steering angle based on the yellow line
            steering_angle = self.follow_yellow_lane(edges)
            # checking whether to turn the steering based on the previous steering angle and yellow line
            steering_angle = self.check_car_turn(steering_angle)
            # setting up the speed and steering angle
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)
            # Display the output
            output = np.dstack((edges, edges, edges))
            cv2.imshow("output", output)
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
            # self.car_turning_angle = self.RIGHT_TURN
            self.car_turning_angle = 0.25
            self.turn_time = 100
        self.prev_steering_angle = steering_angle
        return steering_angle
            

robot = MyRobot()
robot.run()



