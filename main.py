# =========================================
# Professional Handwritten Digit Recognition
# CNN + Better Accuracy + Better Prediction
# Google Colab Version
# =========================================

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab import files

# =========================================
# 1) تحميل Dataset
# =========================================

print("Loading MNIST Dataset...")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Dataset Loaded Successfully ✅")


# =========================================
# 2) تجهيز البيانات بشكل أفضل
# =========================================

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


# =========================================
# 3) بناء نموذج CNN احترافي
# =========================================

model = keras.Sequential([

    keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),

    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    keras.layers.Flatten(),

    keras.layers.Dense(
        128,
        activation='relu'
    ),

    # تقليل Overfitting
    keras.layers.Dropout(0.3),

    keras.layers.Dense(
        10,
        activation='softmax'
    )
])


# =========================================
# 4) Compile
# =========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training CNN Model... Please wait ⏳")


# =========================================
# 5) EarlyStopping
# =========================================

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)


# =========================================
# 6) التدريب
# =========================================

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

print("Model Ready ✅")


# =========================================
# 7) اختبار النموذج
# =========================================

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nFinal Test Accuracy:", accuracy)


# =========================================
# 8) رسم Accuracy و Loss
# =========================================

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.show()


# =========================================
# 9) تجربة على صورة من Dataset
# =========================================

random_index = np.random.randint(0, len(x_test))

plt.imshow(
    x_test[random_index].reshape(28, 28),
    cmap='gray'
)

plt.title("Random Test Image")
plt.axis("off")
plt.show()

prediction = model.predict(
    x_test[random_index].reshape(1, 28, 28, 1),
    verbose=0
)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[random_index])


# =========================================
# 10) رفع صورة من الجهاز والتعرف عليها
# =========================================

print("\nUpload your handwritten digit image:")

uploaded = files.upload()

for filename in uploaded.keys():

    print("Processing:", filename)

    # قراءة الصورة
    img = cv2.imread(
        filename,
        cv2.IMREAD_GRAYSCALE
    )

    # تغيير الحجم
    img = cv2.resize(
        img,
        (28, 28)
    )

    # عكس الألوان
    img = cv2.bitwise_not(img)

    # تقليل التشويش
    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # Threshold أفضل
    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # عرض الصورة بعد المعالجة
    plt.imshow(
        img,
        cmap='gray'
    )

    plt.title("Processed Uploaded Image")
    plt.axis("off")
    plt.show()

    # تجهيز الصورة للنموذج
    img = img.astype("float32") / 255.0

    img = img.reshape(
        1,
        28,
        28,
        1
    )

    # Prediction
    prediction = model.predict(
        img,
        verbose=0
    )

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print("Predicted Digit:", predicted_digit)
    print("Confidence: {:.2f}%".format(confidence))
