#  Fire Detection System using CNN

Một hệ thống phát hiện lửa thời gian thực (Real-time) qua Webcam sử dụng mạng trí tuệ nhân tạo CNN, tự động phát còi báo động và lưu lại hình ảnh bằng chứng khi phát hiện có đám cháy.

---

##  Tổng quan dự án (Project Overview)

Hệ thống phát hiện lửa bao gồm:
- Nhận diện ngọn lửa chính xác từ luồng stream của Webcam/Camera.
- Kích hoạt âm thanh cảnh báo khi độ tự tin (confidence) vượt ngưỡng.
- Tự động chụp và lưu trữ hình ảnh ngọn lửa vào thư mục hệ thống để làm bằng chứng.

Dự án này có thể ứng dụng trong:
- Hệ thống camera giám sát nhà xưởng, kho bãi.
- Cảnh báo cháy sớm trong gia đình.
- Tích hợp vào các robot tuần tra phòng cháy chữa cháy.

---

## 🧠 Nguyên lý hoạt động (How the System Works)

### 🔹 Thu thập & Tiền xử lý dữ liệu
Dữ liệu hình ảnh được chia làm hai nhãn: có lửa (`fire`) và không có lửa (`non_fire`), sau đó được chuẩn hóa kích thước để đưa vào mạng CNN.

### 🔹 Mô hình CNN (Convolutional Neural Network)
Hệ thống sử dụng mạng trích xuất đặc trưng tích chập (CNN) được xây dựng bằng TensorFlow/Keras nhằm nhận diện các pattern của ngọn lửa (màu sắc, hình dáng, độ sáng).

### 🔹 Xử lý thời gian thực với OpenCV
- Đọc luồng video từ Webcam theo từng khung hình (frame).
- Đưa khung hình qua mô hình dự đoán. Nếu kết quả là lửa, hệ thống lập tức kích hoạt hàm lưu ảnh và phát âm thanh.

---

##  Công nghệ sử dụng (Technologies Used)

- **Python**: Ngôn ngữ lập trình chính.
- **TensorFlow / Keras**: Xây dựng, huấn luyện mô hình Deep Learning (CNN).
- **OpenCV**: Xử lý hình ảnh, quản lý luồng video từ Webcam.
- **NumPy & Matplotlib**: Xử lý mảng dữ liệu và vẽ biểu đồ đánh giá mô hình.

---

##  Cấu trúc dự án (Project Structure)

```text
HungML/
├── datasetfire/
│   ├── fire/                # Thư mục chứa ảnh có lửa
│   └── non_fire/            # Thư mục chứa ảnh không có lửa
├── train.py                 # File script huấn luyện mô hình
├── demo.py                  # File script chạy demo hệ thống qua webcam
├── fire_modelv4.h5          # File lưu trữ mô hình đã huấn luyện xong
├── detected/                # Thư mục lưu ảnh tạm thời
└── real_fire/               # Thư mục lưu ảnh khi phát hiện lửa thật
