import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_data(data_dir, classes, img_size=100, max_per_class=1500):
    X, y = [], []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        img_names = os.listdir(class_path)[:max_per_class]

        for img_name in img_names:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.equalizeHist(img)

            X.append(img)
            y.append(label)

    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    y = to_categorical(y, num_classes=len(classes))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test
