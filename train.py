import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


class Data:
    def __init__(self, df):
        self.df = df
        self.numeric_cols = self.df.select_dtypes(include=["float64"]).columns
        self.categorical_cols = self.df.select_dtypes(include=["object"]).columns

    def _normalize_numeric(self):
        norm_model = preprocessing.MinMaxScaler().fit(self.df[self.numeric_cols])
        self.df[self.numeric_cols] = norm_model.transform(self.df[self.numeric_cols])
        pickle.dump(norm_model, open("norm_model.pkl", "wb"))

    def _normalize_categorical(self):
        self.df = pd.concat(
            [self.df, pd.get_dummies(self.df[self.categorical_cols], dtype=int)], axis=1
        )
        dummies_cols = pd.get_dummies(self.df[self.categorical_cols], dtype=int).columns
        self.df.drop(self.categorical_cols, axis=1, inplace=True)
        self.categorical_cols = dummies_cols

    def normalize(self):
        self._normalize_numeric()
        self._normalize_categorical()


def calculate_accuracy(model, X, y):
    y_pred = model.predict(X)
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = pd.get_dummies(y).values.argmax(axis=1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(y_true_labels, y_pred_labels), tf.float32)
    )
    print(f"Accuracy: {accuracy:.4f}")


def create_model(input_shape, num_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=input_shape),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(X, y, num_classes):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_encoded = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    input_shape = (X_train.shape[1],)
    model = create_model(input_shape, num_classes)
    model.fit(
        X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test)
    )

    print(model.summary())
    
    people = pd.DataFrame(label_encoder.classes_, columns=["username"])

    people.to_csv("people.csv", index=False)

    return model


def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved successfully at {model_path}")


def main():
    df = pd.read_csv("data.csv")
    data = Data(df)
    data.normalize()

    # data.df.to_csv("normalized_data.csv", index=False)

    X = data.df.filter(regex="^(?!.*username).*$")
    y = data.df.filter(regex="^username_", axis=1).idxmax(axis=1)

    num_classes = len(y.unique())
    model = train_model(X, y, num_classes)

    calculate_accuracy(model, X, y)

    save_model(model, "model")


if __name__ == "__main__":
    main()
