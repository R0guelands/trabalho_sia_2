import tensorflow as tf
import pandas as pd
import cv2
import mediapipe as mp
import pandas as pd
import pickle
import numpy as np


def draw_facemesh(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image,
        face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
    )


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def infer(model, data):
    data = data.to_numpy()
    predictions = model.predict(data)
    return tf.argmax(predictions, axis=1).numpy()[0]


def normalize(df):
    columns = df.columns
    norm_model = pickle.load(open("norm_model.pkl", "rb"))
    return pd.DataFrame(norm_model.transform(df), columns=columns)


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def calculate_distances(face_landmarks):
    num_points = len(face_landmarks.landmark)
    distances_1d = {}
    for i in range(num_points - 1):
        x1, y1, z1 = (
            face_landmarks.landmark[i].x,
            face_landmarks.landmark[i].y,
            face_landmarks.landmark[i].z,
        )
        x2, y2, z2 = (
            face_landmarks.landmark[i + 1].x,
            face_landmarks.landmark[i + 1].y,
            face_landmarks.landmark[i + 1].z,
        )
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        distances_1d[f"distance_{i}"] = distance

    return pd.DataFrame(distances_1d, index=[0])


def main():
    model_path = "model"
    model = load_model(model_path)

    users = pd.read_csv("people.csv")
    users = users["username"].apply(lambda x: x.split("_")[1]).to_list()

    cap = cv2.VideoCapture("http://192.168.18.118:8080/video", cv2.CAP_FFMPEG)

    df = pd.DataFrame()
    result = "Unknown"
    with mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # draw_facemesh(image, face_landmarks)
                    df = pd.concat(
                        [df, calculate_distances(face_landmarks)], ignore_index=True
                    )
                    if df.shape[0] == 64:
                        df = normalize(df)
                        result = infer(model, df)
                        df = pd.DataFrame()
                    cv2.putText(
                        image,
                        f"Person: {users[result] if result != 'Unknown' else result}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )

            cv2.imshow("Face Mesh", image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
