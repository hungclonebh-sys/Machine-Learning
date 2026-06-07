# Machine-Learning
Project

#Hệ thống phát hiện lửa bằng CNN
Giới thiệu
Hệ thống phát hiện lửa thời gian thực qua webcam, phát còi báo động và lưu ảnh khi có cháy.

#Công nghệ
Python + TensorFlow/Keras (CNN)
OpenCV (xử lý ảnh, webcam)
NumPy, Matplotlib

# Cấu trúc dự án
HungML/
├── datasetfire/fire/          # Ảnh có lửa
├── datasetfire/non_fire/      # Ảnh không lửa
├── train.py                   # Huấn luyện
├── demo.py                    # Chạy demo
├── fire_modelv4.h5            # Mô hình đã train
├── detected/                  # Ảnh tạm
└── real_fire/                 # Ảnh lửa thật


#Cài đặt
bash
pip install tensorflow opencv-python numpy matplotlib


#Cách chạy
bash
python codetrain.py    # Huấn luyện
python codedemo.py     # Chạy demo


#Điều khiển demo
Phím	Chức năng
Q	Thoát
S	Chụp ảnh
F	Lọc ảnh
+/-	Tăng/giảm ngưỡng


#Kết quả
Độ chính xác: 93.8%


#Phát hiện được: cháy rừng, bật lửa, nến, bếp gas


#Hạn chế
Đôi khi nhầm với vật màu cam/đỏ/môi trường quá sáng


#Chậm trên CPU (10-20fps)


#Hướng phát triển rộng hơn
MobileNetV2 để chạy nhanh hơn
YOLOv8 để khoanh vùng lửa
Chạy trên Raspberry Pi

#Tác giả
- [Họ tên bạn]

