Bước 1: Load dữ liệu từ file CSV

Bước 2: Chia tập dữ liệu thành 2 phần train và test theo tỉ lệ 8:2

Bước 3: Chuyển đổi các đặc trưng dạng phân loại sang dạng số sử dụng OneHotEncoder

Bước 4: Tạo mô hình Linear Regression với sklearn và huấn luyện trên dữ liệu train

Bước 5: Sử dụng mô hình đã huấn luyện để dự đoán trên tập test

Bước 6: Đánh giá hiệu quả của mô hình trên cả 2 tập train và test