import tensorflow as tf
import pandas as pd
import cv2
import mediapipe as mp
import pandas as pd
import pickle
import numpy as np
import time

def draw_facemesh(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image,
        face_landmarks,
        mp_face_mesh.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
    )


def load_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)


def infer(model: tf.keras.Model, data: pd.DataFrame):
    data = data.to_numpy()
    predictions = model.predict(data, use_multiprocessing=True)
    probabilities = np.max(predictions, axis=1)
    predicted_class = np.argmax(predictions, axis=1)
    chosen_class = np.bincount(predicted_class).argmax()

    percentage = probabilities[chosen_class] * 100

    if percentage > 65:
        return chosen_class, percentage
    else:
        return "Unknown", percentage


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

def get_bounding_box(face_landmarks, image_width, image_height):
    x_min = image_width
    y_min = image_height
    x_max = 0
    y_max = 0
    
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
        if y < y_min:
            y_min = y
        if y > y_max:
            y_max = y
    
    width = x_max - x_min
    height = y_max - y_min
    
    return x_min, y_min, width, height

# Função para desenhar o retângulo e inserir o nome da pessoa
def draw_rectangle(image, bbox, name):
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        image,
        f"Person: {name}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        3,
    )
    

def main():
    start_time = time.time()

    model_path = "model"
    model = load_model(model_path)

    users = pd.read_csv("people.csv")
    users = users["username"].apply(lambda x: x.split("_")[1]).to_list()

    # cap = cv2.VideoCapture("http://192.168.18.118:8080/video", cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(1)

    df = pd.DataFrame()
    result = "Unknown"
    percentage = 0
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
                    if df.shape[0] == 30:
                        df = normalize(df)
                        result, percentage = infer(model, df)
                        df = pd.DataFrame()
                        
                    bbox = get_bounding_box(face_landmarks, image.shape[1], image.shape[0])
                    name = users[result] if result != 'Unknown' else result
                    draw_rectangle(image, bbox, name)
                    
            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(image, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

            cv2.imshow("Face detection", image)

            start_time = time.time()

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
