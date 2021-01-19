Detect Information Identity Card
Trích xuất thông tin từ ảnh chụp chứng minh nhân dân.
Hướng dẫn cài đặt:
- Clone code từ github: https://github.com/quanghai1279/Detect_Info_Identity_Card.git
- Cài đặt Python 3.7: https://www.python.org/downloads/release/python-370/
- Cài đặt PyCharm: https://www.jetbrains.com/pycharm/download/
- Tạo project mới trong PyCharm với Location là thư mục code đã clone từ github.
- Chọn File -> Setting... -> Project -> Python Structure và đánh dấu thư mục src là Sources.
- Chọn File -> Setting... -> Project -> Python Interpreter và cài các thư viện:
    + opencv-python 	: 4.4.0.44
    + numpy 		: 1.17.2
    + Shapely 		: 1.7.1
    + tensorflow 	: 1.15.0
    + Keras		: 2.1.3
    + Pillow		: 7.2.0
    + editdistance	: 0.5.3
    + h5py		: 2.10.0
- Chạy file rnn_training.py để huấn luyện mô hình.
- Chạy file main.py để chạy ứng dụng với đầu vào là ảnh, đầu ra là thông tin nhận diện được