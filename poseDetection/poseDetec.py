import cv2
import mediapipe as mp
import numpy as np

# initialize mediapipe pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# input video for the pose detection
capture = cv2.VideoCapture('C:\\testCV\\vid2.MP4')
# cap = cv2.VideoCapture(0)  ##----for live camera


# read each frame from capture object
while True:
    ret, image = capture.read()
    image = cv2.resize(image, (600, 600))  # resize the image frame for your screen

    # perform pose detection
    results = pose.process(image)
    # draw the detectedpose on the input video
    mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    cv2.imshow("Pose Estimation", image)  # display the pose on the input video

    # Extract and pose on plain image
    h, w, c = image.shape  # get shape of the input frame
    opImg = np.zeros([h, w, c])  # create a blank image with input frame size
    opImg.fill(255)  # set the background to white

    # draw the extracted pose on white frame
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    # display extracted pose
    cv2.imshow("Extracted Pose", opImg)

    # print landmarks
    print(results.pose_landmarks)

    cv2.waitKey(1)
