import numpy as np

# Định nghĩa hàm dự đoán
def h(w, x):
    return np.sign(np.dot(w.T, x))

# Khởi tạo dữ liệu từ đề bài (x đã có bias)
w = np.array([[-2], [1], [0]])
x = np.array([[2], [3], [1]])
y = 1

# 1. Kiểm tra mẫu có bị phân lớp sai hay không
wTx = np.dot(w.T, x)[0, 0]
pred = h(w, x)[0, 0]
is_misclassified = (pred != y)

print("=== 1. KIỂM TRA MẪU PHÂN LỚP ===")
print(f"w^T * x = {wTx}")
print(f"Nhãn dự đoán h(x) = {pred}")
print(f"Nhãn thực tế y    = {y}")

if is_misclassified:
    print("=> Kết luận: Mẫu BỊ PHÂN LỚP SAI.")
else:
    print("=> Kết luận: Mẫu được phân lớp đúng.")

# 2. Thực hiện cập nhật Perceptron nếu bị sai
if is_misclassified:
    # Công thức cập nhật trong perceptron: w_new = w + y * x
    w_new = w + y * x
    print("\n=== 2. CẬP NHẬT TRỌNG SỐ PERCEPTRON ===")
    print("Vector trọng số mới w_new:")
    print(w_new)

    # 3. Tính lại giá trị w^T * x sau cập nhật
    w_new_Tx = np.dot(w_new.T, x)[0, 0]
    pred_new = h(w_new, x)[0, 0]
    
    print("\n=== 3. TÍNH LẠI SAU CẬP NHẬT ===")
    print(f"w_new^T * x = {w_new_Tx}")
    print(f"Nhãn dự đoán mới h_new(x) = {pred_new}")