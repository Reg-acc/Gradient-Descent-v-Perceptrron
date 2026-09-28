import numpy as np

# 1. Khai báo dữ liệu theo yêu cầu đề bài
w = np.array([[1], [2], [-10]])
x = np.array([[3], [4], [1]])
y = -1  # Nhãn thực tế

# Hàm dự đoán nhãn h(w, x)
def h(w, x):
    return np.sign(np.dot(w.T, x))

# -------------------------------------------------------------
# Câu 1: Tính w^T * x
wTx = np.dot(w.T, x)[0, 0]
print(f"1. Tích w^T * x = {wTx}")

# Câu 2: Xác định nhãn dự đoán h(w, x)
y_pred = int(h(w, x)[0, 0])
print(f"2. Nhãn dự đoán (y_hat) = {y_pred}")

# Câu 3: Kiểm tra phân lớp sai với y = -1
is_misclassified = (y_pred != y)
print(f"3. Nhãn thực tế y = {y}")
if is_misclassified:
    print("   -> Kết luận: Điểm dữ liệu BỊ PHÂN LỚP SAI.")
else:
    print("   -> Kết luận: Điểm dữ liệu ĐƯỢC PHÂN LỚP ĐÚNG.")