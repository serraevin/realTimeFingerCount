import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(maxHands=1, detectionCon=0.8)
video = cv2.VideoCapture(0)  # Use index 0 for the default camera

# Load the images for different hand gestures
fingers_up_images = [
    cv2.imread("image_path_for_0_fingers_up.jpg"),
    cv2.imread("image_path_for_1_finger_up.jpg"),
    cv2.imread("image_path_for_2_fingers_up.jpg"),
    cv2.imread("image_path_for_3_fingers_up.jpg"),
    cv2.imread("image_path_for_4_fingers_up.jpg"),
    cv2.imread("image_path_for_5_fingers_and_thumbs_up.jpg")
]

while True:
    ret, img = video.read()
    img = cv2.flip(img, 1)
    hands, _ = detector.findHands(img)

    if hands and hands[0].lmList:
        hand_landmarks = hands[0].lmList
        finger_up = detector.fingersUp(hand_landmarks)

        # Select the appropriate image based on the number of fingers
        num_fingers = finger_up.count(1)
        gesture_image = fingers_up_images[num_fingers]

        # Resize the gesture image to fit the hand region
        gesture_image = cv2.resize(gesture_image, (220, 280))

        # Replace the region of interest (hand region) with the gesture image
        x1, y1 = hand_landmarks[17][1], hand_landmarks[17][2]
        x2, y2 = hand_landmarks[5][1], hand_landmarks[5][2]
        img[y1:y1 + 280, x1:x1 + 220] = gesture_image

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
