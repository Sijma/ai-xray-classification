import keras
import numpy as np
import tensorflow as tf
from keras import models, layers

DATA_DIR = "./COVID-19_Radiography_Dataset"
DISPLAY_CONFUSION_MATRICES = False

if DISPLAY_CONFUSION_MATRICES:
    import matplotlib.pyplot as plt
    import seaborn as sns


class ProcessedDataset:
    def __init__(self, devel_ds, train_ds, val_ds, test_ds, class_names):
        self.devel_ds = devel_ds
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.class_names = class_names


# Returns the input number bound to min or max provided, if the number surpasses them.
def clamp(number, min_num, max_num):
    return max(min(number, max_num), min_num)


def confusion_matrix(model, test_ds):
    y_test = []
    y_pred = []
    for x_1, y_1 in test_ds:
        y_pred_1 = model.predict(x_1)
        y_test.append(y_1)
        y_pred.append(y_pred_1)
    y_true = np.concatenate(y_test)
    y_p = np.concatenate(y_pred)
    y_hat = tf.argmax(y_p, axis=1)
    cm = tf.math.confusion_matrix(y_true, y_hat)
    return cm, y_hat


def prepare_datasets(data_dir, train_pct=0.6, val_pct=0.2, test_pct=0.2, batch_size=64, img_size=(299, 299)):
    train_pct = clamp(train_pct, 0, 1)
    val_pct = clamp(val_pct, 0, 1)
    test_pct = clamp(test_pct, 0, 1)

    if train_pct + val_pct + test_pct > 1:
        raise NameError('Invalid dataset splits. Total should be <= 1.')

    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size)

    dataset_size = len(full_ds)
    print("dataset_size = ", dataset_size)

    train_size = int(train_pct * dataset_size)
    val_size = int(val_pct * dataset_size)
    test_size = int(test_pct * dataset_size)

    devel_ds = full_ds.take(train_size+val_size)
    train_ds = devel_ds.take(train_size)
    val_ds = devel_ds.skip(train_size)
    test_ds = full_ds.skip(train_size+val_size)

    # Redundant if we want to use the entire database, we could assume test takes everything that's left.
    test_ds = test_ds.take(test_size)

    class_names = full_ds.class_names

    return ProcessedDataset(devel_ds, train_ds, val_ds, test_ds, class_names)


def cnn1(num_classes):
    new_model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(299, 299, 3)),
        layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return new_model


def cnn2(num_classes):
    new_model = models.Sequential([
        layers.Rescaling(1. / 255, input_shape=(299, 299, 3)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=4),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(strides=2),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return new_model


def resized_EfficientNetB0():
    resize = tf.keras.Sequential([layers.Resizing(224, 224)])
    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = resize(inputs)
    outputs = tf.keras.applications.efficientnet.EfficientNetB0()(x)
    inference_model = tf.keras.Model(inputs, outputs)
    return inference_model


def execute_cnn(model, dataset, epochs, batch_size):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=(10 ** -3), beta_1=0.9, beta_2=0.99)

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(dataset.train_ds, validation_data=dataset.val_ds, epochs=epochs, batch_size=batch_size,
              callbacks=[callback])

    display_confusion_matrix(model, dataset.test_ds, dataset.class_names)


def display_confusion_matrix(model, test_ds, class_names):
    if not DISPLAY_CONFUSION_MATRICES:
        return

    cm = confusion_matrix(model, test_ds)[0]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('True Value')
    plt.show()


def simple_cnn(dataset):
    epochs = 20
    batch_size = 64

    model = cnn1(len(dataset.class_names))

    execute_cnn(model, dataset, epochs, batch_size)


def deeper_cnn(dataset):
    epochs = 20
    batch_size = 64

    model = cnn2(len(dataset.class_names))

    execute_cnn(model, dataset, epochs, batch_size)


def trained_cnn(dataset):
    epochs = 5
    batch_size = 32

    model = resized_EfficientNetB0()

    execute_cnn(model, dataset, epochs, batch_size)


if __name__ == '__main__':
    dataset = prepare_datasets(DATA_DIR)

    simple_cnn(dataset)
    deeper_cnn(dataset)
    trained_cnn(dataset)
