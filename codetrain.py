import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import os

# ===== CẤU HÌNH =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATASET = r"D:\HungML\datasetfire"
MODEL_PATH = r"D:\HungML\fire_modelv5.h5"

# ===== KIỂM TRA DATASET =====
print("Kiem tra dataset:")
for folder in os.listdir(DATASET):
    path = os.path.join(DATASET, folder)
    if os.path.isdir(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  {folder}: {count} anh")

# ===== CHUẨN BỊ DỮ LIỆU =====
print("\nChuan bi du lieu...")

train_data = image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_data = image_dataset_from_directory(
    DATASET,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

print(f"Classes: {train_data.class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(AUTOTUNE)
val_data = val_data.cache().prefetch(AUTOTUNE)

# ===== XÂY DỰNG MODEL =====
print("\nXay dung model...")

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ===== HUẤN LUYỆN =====
print("\nBat dau train...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=callbacks)

# ===== ĐÁNH GIÁ =====
val_loss, val_acc = model.evaluate(val_data)
print(f"\nDo chinh xac: {val_acc*100:.2f}%")
print(f"Model luu tai: {MODEL_PATH}")

# ===== VẼ ĐỒ THỊ =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Validation")
ax1.set_title("Do chinh xac")
ax1.legend()
ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Validation")
ax2.set_title("Loss")
ax2.legend()
plt.tight_layout()
plt.savefig(r"D:\HungML\training_result.png")
plt.show()

print("Hoan tat!")
