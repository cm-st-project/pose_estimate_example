import cv2
import mediapipe as mp

# Initialize MediaPipe Pose solution
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Initialize MediaPipe drawing utilities
mpDraw = mp.solutions.drawing_utils

# Open video capture (comment/uncomment appropriate line for webcam or video file)
# cap = cv2.VideoCapture(0)  # Use webcam
cap = cv2.VideoCapture('video.mp4')  # replace video.mp4 with your video
pTime = 0
width = 800
height = 600
while True:
    # Read a frame from the video capture
    success, img = cap.read()

    img = cv2.resize(img, (width, height))
    if not success:
        break

    # Convert BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image with the pose estimation model
    results = pose.process(imgRGB)

    # Print the list of landmarks (x, y, z, visibility)
    print(results.pose_landmarks)

    if results.pose_landmarks:
        # Draw the pose landmarks and connections on the image
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # Draw circles at the vertices of each pose landmark
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Display the annotated image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
