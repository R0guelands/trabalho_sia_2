import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


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

    def _balance_column(
        self, df: pd.DataFrame, col_name: str, option: str = "up"
    ) -> pd.DataFrame:
        X = df.drop(col_name, axis=1)
        y = df[col_name]

        if option == "up":
            sampler = RandomOverSampler()
        elif option == "down":
            sampler = RandomUnderSampler()

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=col_name)

        return pd.concat([X_resampled, y_resampled], axis=1)

    def balance(self, col_list=None, option: str = "up"):
        if col_list is not None:
            df_categorical = self.df[col_list].select_dtypes(include=["object"])
        else:
            df_categorical = self.df.select_dtypes(include=["object"])

        if len(df_categorical.columns) == 0:
            return

        rebalanced_df = self.df.copy()
        for col_name in df_categorical.columns:
            rebalanced_df = self._balance_column(rebalanced_df, col_name, option)

        self.df = rebalanced_df.reset_index(drop=True).copy()


def calculate_accuracy(model, X, y):
    y_pred = model.predict(X)
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = pd.get_dummies(y).values.argmax(axis=1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(y_true_labels, y_pred_labels), tf.float32)
    )
    print(f"Accuracy: {accuracy:.4f}")

# def create_model(input_shape, num_classes):
#     l2_reg = 0.015
#     # model = tf.keras.models.Sequential(
#     #     [
#     #         tf.keras.layers.Dense(1024, activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(2048, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(num_classes, activation="softmax"),
#     #     ]
#     # )
#     model = tf.keras.models.Sequential(
#         [
#             tf.keras.layers.Dense(
#                 512,
#                 activation="relu",
#                 input_shape=input_shape,
#                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
#             ),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation="relu",
#                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
#             ),
#             tf.keras.layers.Dense(
#                 512,
#                 activation="relu",
#                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
#             ),
#             tf.keras.layers.Dense(
#                 1024,
#                 activation="relu",
#                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
#             ),
#             tf.keras.layers.Dense(
#                 512,
#                 activation="relu",
#                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
#             ),
#             tf.keras.layers.Dense(num_classes, activation="softmax"),
#         ]
#     )
#     # model = tf.keras.models.Sequential(
#     #     [
#     #         tf.keras.layers.Dense(256, activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
#     #         tf.keras.layers.Dense(num_classes, activation="softmax"),
#     #     ]
#     # )
#     model.compile(
#         optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
#     )
#     return model

class MyHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        l2_reg = hp.Choice('l2_reg', values=[0.001, 0.01, 0.1])
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(
            units=hp.Int('units_1', min_value=128, max_value=2048, step=128),
            activation='relu',
            input_shape=self.input_shape,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))

        for i in range(hp.Int('num_layers', min_value=1, max_value=4)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int('units_' + str(i+2), min_value=128, max_value=2048, step=128),
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            ))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
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
    
    tuner = RandomSearch(
        MyHyperModel(input_shape, num_classes),
        objective='val_accuracy',
        max_trials=10,
        directory='tuning_results',
        project_name='face_regonizer'
    )
    
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Fit the model with the best parameters
    best_model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    people = pd.DataFrame(label_encoder.classes_, columns=["username"])

    people.to_csv("people.csv", index=False)

    return best_model

# def train_model(X, y, num_classes):
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
#     y_encoded = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_encoded, test_size=0.2, random_state=42
#     )
#     input_shape = (X_train.shape[1],)
#     model = create_model(input_shape, num_classes)
#     print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
#     model.fit(
#         X_train,
#         y_train,
#         epochs=200,
#         batch_size=64,
#         validation_data=(X_test, y_test),
#         use_multiprocessing=True,
#     )

#     print(model.summary())

#     people = pd.DataFrame(label_encoder.classes_, columns=["username"])

#     people.to_csv("people.csv", index=False)

#     return model


def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved successfully at {model_path}")


def main():
    df = pd.read_csv("data.csv")
    data = Data(df)
    data.normalize()
    data.balance()

    # data.df.to_csv("normalized_data.csv", index=False)

    X = data.df.filter(regex="^(?!.*username).*$")
    y = data.df.filter(regex="^username_", axis=1).idxmax(axis=1)

    num_classes = len(y.unique())
    model = train_model(X, y, num_classes)

    calculate_accuracy(model, X, y)

    save_model(model, "model")


if __name__ == "__main__":
    main()
