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

##  Nguyên lý hoạt động (How the System Works)

### 🔹 Thu thập & Tiền xử lý dữ liệu
- Dữ liệu hình ảnh được chia làm hai nhãn: có lửa (`fire`) - 1600 image và không có lửa (`non_fire`) - 1400 image , sau đó được chuẩn hóa kích thước để đưa vào mạng CNN.
- Datasetfire:
- <img width="1339" height="666" alt="image" src="https://github.com/user-attachments/assets/b24cb575-9209-482b-8235-643a2338cde9" />
- Datasetfire tự tạo:
- <img width="1352" height="493" alt="image" src="https://github.com/user-attachments/assets/e59d0749-6994-4377-82ca-be68415af12b" />
- Datasetfire non_fire:
- <img width="1663" height="711" alt="image" src="https://github.com/user-attachments/assets/459a613a-406b-4857-a721-dc48f692fdd5" />



### 🔹 Mô hình CNN (Convolutional Neural Network)
Hệ thống sử dụng mạng trích xuất đặc trưng tích chập (CNN) được xây dựng bằng TensorFlow/Keras nhằm nhận diện các pattern của ngọn lửa (màu sắc, hình dáng, độ sáng).

### 🔹 Xử lý thời gian thực với OpenCV
- Đọc luồng video từ Webcam theo từng khung hình (frame).
- Đưa khung hình qua mô hình dự đoán. Nếu kết quả là lửa, hệ thống lập tức kích hoạt hàm lưu ảnh và phát âm thanh.
- <img width="1281" height="507" alt="image" src="https://github.com/user-attachments/assets/ceaf2d64-df9b-4dd6-90d0-ff304bc787e0" />


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
