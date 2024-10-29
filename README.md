# Import required libraries
import cv2
import time
import mediapipe as mp

# Initialize the Mediapipe Holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize the Holistic model with parameters
holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open video capture (0) for the default camera
capture = cv2.VideoCapture(0)

# Initialize time variables for FPS calculation
previousTime = 0
currentTime = 0

while capture.isOpened():
    # Capture frame-by-frame
    ret, frame = capture.read()
    if not ret:
        break

    # Resize the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Convert the BGR frame to RGB for processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Disable writing to the image to optimize processing time
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Convert back to BGR for OpenCV operations
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw facial landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        )

    # Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Calculate and display FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(image, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Press 'q' to exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
holistic_model.close()

# Print available landmark points in hand
for landmark in mp_holistic.HandLandmark:
    print(landmark, landmark.value)

# Example of accessing wrist landmark value
print("Wrist landmark value:", mp_holistic.HandLandmark.WRIST.value)
