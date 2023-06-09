from draw_holistic_and_extract_points import *
import numpy as np
import cv2
from keras.models import load_model


actions = np.array(['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н',
                    'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь',
                    'Э', 'Ю', 'Я'])
model = load_model('rsl_model4.h5')
# colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
#
#
# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#     return output_frame


# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        H, W, _ = frame.shape

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks_specified(image, results)

        # 2. Prediction logic
        if mp_holistic.HandLandmark:
            keypoints = extract_keypoints_1hand(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)] + ' ' + str(max(res)))
                predictions.append(np.argmax(res))
                #
                # # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(str(actions[np.argmax(res)]))
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                # image = prob_viz(res, dataCollection.actions, image, colors)

                cv2.rectangle(image, (0, 0), (640, 40), (40, 40, 40), 1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

    cap.release()
    cv2.destroyAllWindows()
