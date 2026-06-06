import cv2
import numpy as np
import tensorflow as tf
import datetime
import os
import winsound
import shutil
import time

MODEL_PATH = r"D:\HungML\fire_modelv4.h5"
SAVE_DIR = r"D:\HungML\detected"
REAL_FIRE_DIR = r"D:\HungML\real_fire"
THRESHOLD = 0.5
ALARM_TIMES = 3
AUTO_CLEAN_INTERVAL = 1800
MIN_IMAGES_TO_FILTER = 10

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(REAL_FIRE_DIR, exist_ok=True)

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast

def filter_by_contrast():
    print("\nDang phan loai theo do tuong phan...")
    
    images = [f for f in os.listdir(SAVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print("Khong co anh nao")
        return
    
    contrast_list = []
    for img_file in images:
        img_path = os.path.join(SAVE_DIR, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            contrast = calculate_contrast(img)
            contrast_list.append((img_file, img_path, contrast))
    
    if len(contrast_list) == 0:
        print("Khong doc duoc anh nao")
        return
    
    contrast_list.sort(key=lambda x: x[2], reverse=True)
    top_count = max(1, len(contrast_list) // 2)
    top_images = contrast_list[:top_count]
    
    moved_count = 0
    for img_file, img_path, contrast in top_images:
        dest = os.path.join(REAL_FIRE_DIR, img_file)
        shutil.move(img_path, dest)
        moved_count += 1
        print(f"DA CHON: {img_file} (do tuong phan: {contrast:.2f})")
    
    for img_file, img_path, contrast in contrast_list[top_count:]:
        os.remove(img_path)
        print(f"DA XOA: {img_file} (do tuong phan: {contrast:.2f})")
    
    print(f"\nDa chon {moved_count} anh co do tuong phan cao nhat vao real_fire")

print("Dang tai model...")
if not os.path.exists(MODEL_PATH):
    print("Khong tim thay model")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
print("Model da tai xong")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Khong the mo webcam")
    exit()

print("Nhan Q de thoat | S de chup | F de phan loai | R de xoa tat ca")

fire_count = 0
alarm_playing = False
frame_count = 0
saved_count = 0
pred = 0.0
last_clean_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img, axis=0).astype("float32")
    pred = model.predict(img_array, verbose=0)[0][0]

    is_fire = pred > THRESHOLD
    conf = pred * 100 if is_fire else (1 - pred) * 100

    if is_fire:
        fire_count += 1
    else:
        fire_count = 0
        alarm_playing = False

    display = frame.copy()
    h, w = display.shape[:2]

    if is_fire:
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 255), 5)
        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 200), -1)
        cv2.putText(display, f"FIRE! ({conf:.1f}%)", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

        if fire_count >= 3 and not alarm_playing:
            for _ in range(ALARM_TIMES):
                winsound.Beep(1000, 300)
            alarm_playing = True

        if fire_count == 3:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"fire_{ts}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Da luu: {filename}")
            
            pending = len([f for f in os.listdir(SAVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if pending >= MIN_IMAGES_TO_FILTER:
                print(f"Du {MIN_IMAGES_TO_FILTER} anh, tien hanh phan loai...")
                filter_by_contrast()

    else:
        cv2.rectangle(display, (0, 0), (w, h), (0, 200, 0), 3)
        cv2.rectangle(display, (0, 0), (w, 70), (0, 150, 0), -1)
        cv2.putText(display, f"SAFE ({conf:.1f}%)", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)

    cv2.rectangle(display, (0, h - 40), (w, h), (40, 40, 40), -1)
    now = datetime.datetime.now().strftime("%H:%M:%S")
    pending = len([f for f in os.listdir(SAVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    cv2.putText(display, f"{now} | Saved: {saved_count} | Pending: {pending} | Thresh: {THRESHOLD}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    cv2.imshow("Fire Detection", display)

    current_time = time.time()
    if current_time - last_clean_time >= AUTO_CLEAN_INTERVAL:
        pending = len([f for f in os.listdir(SAVE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if pending > 0:
            filter_by_contrast()
        last_clean_time = current_time

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"manual_{ts}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f"Da chup: {filename}")
    elif key == ord("f"):
        filter_by_contrast()
        last_clean_time = current_time
    elif key == ord("r"):
        for f in os.listdir(SAVE_DIR):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.remove(os.path.join(SAVE_DIR, f))
        print("Da xoa tat ca")
    elif key == ord("+"):
        THRESHOLD = min(0.95, THRESHOLD + 0.05)
        print(f"Threshold: {THRESHOLD}")
    elif key == ord("-"):
        THRESHOLD = max(0.05, THRESHOLD - 0.05)
        print(f"Threshold: {THRESHOLD}")

cap.release()
cv2.destroyAllWindows()

print(f"\nTong ket:")
print(f"  Anh da luu: {saved_count}")
print(f"  Anh trong real_fire: {len(os.listdir(REAL_FIRE_DIR)) if os.path.exists(REAL_FIRE_DIR) else 0}")
