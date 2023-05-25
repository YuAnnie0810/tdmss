from data_collection import actions
from draw_holistic_and_extract_points import *
import numpy as np
import cv2
from keras.models import load_model

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
        print(str(H) + ' ' + str(W))

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

            if len(sequence) == 30:
                x_ = []
                y_ = []
                for landmark in sequence:
                    x_1 = [value for key, value in enumerate(landmark, 1) if key % 3 != 0]
                    x_x = [value for key, value in enumerate(x_1, 1) if key % 3 != 0]
                    y_y = [value for key, value in enumerate(x_, 1) if key % 2 != 0]
                    x_ = np.concatenate([x_, x_x])
                    y_ = np.concatenate([y_, y_y])

                x1 = int(min(x_) * W)
                y1 = int(min(y_) * H)
                x2 = int(max(x_) * W) - 20
                y2 = int(max(y_) * H) - 20

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
                #
                # if len(sentence) > 5:
                #     sentence = sentence[-5:]

                # Viz probabilities
                # image = prob_viz(res, dataCollection.actions, image, colors)

                # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
                # cv2.putText(image, str(sentence[-1]), (x1, y1),
                #             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if len(sentence) > 0:
                    to_print = str(sentence[-1]) + ' ' + str(max(res))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(image, to_print, (x1, y1),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(image, ''.join(sentence), (x1, y1),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
