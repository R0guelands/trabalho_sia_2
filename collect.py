import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def draw_facemesh(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image,
        face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
    )


def save_to_csv(df):
    username = input("Enter your name: ")
    df["username"] = username.upper()
    database = load_df()
    database = pd.concat([database, df], ignore_index=True)
    database.to_csv("data.csv", index=False)


def load_df():
    return pd.read_csv("data.csv") if os.path.exists("data.csv") else pd.DataFrame()


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
    # cap = cv2.VideoCapture("http://192.168.18.118:8080/video", cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(1)
    
    df = pd.DataFrame()

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
                    draw_facemesh(image, face_landmarks)
                    df = pd.concat(
                        [df, calculate_distances(face_landmarks)], ignore_index=True
                    )

            cv2.imshow("Face Mesh", image)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    save_to_csv(df)


if __name__ == "__main__":
    main()
