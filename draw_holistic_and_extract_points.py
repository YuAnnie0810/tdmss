import mediapipe as mp
import numpy as np
import cv2


mp_hands = mp.solutions.hands #Hands
mp_drawing = mp.solutions.drawing_utils #Drawing lines
mp_holistic = mp.solutions.holistic #Holistic


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks_specified(image, results):
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def draw_styled_landmarks_not_specified(image, results):
    if not results.multi_hand_landmarks:
        return
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


def extract_keypoints_both_hands(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


def extract_keypoints_1hand(results):
    if results.left_hand_landmarks:
        hlm = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten()
    elif results.right_hand_landmarks:
        hlm = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten()
    else:
        hlm = np.zeros(21 * 3)
    return hlm


def extract_keypoints_hand(results):
    hlm = []
    if not results.multi_hand_landmarks:
        return np.zeros(21*3)
    for handLms in results.multi_hand_landmarks:
        for hl_id, lm in enumerate(handLms.landmark):
            hlm = np.append(hlm, [lm.x, lm.y, lm.z])
    return hlm
