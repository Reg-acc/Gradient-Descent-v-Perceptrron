import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, random_state=None):
        """
        Khởi tạo mô hình Perceptron
        :param random_state: Seed ngẫu nhiên để tái lập kết quả
        """
        self.w_ = None          # Lưu vector trọng số cuối cùng (bao gồm cả bias)
        self.w_history_ = []    # Lưu lịch sử trọng số qua từng bước cập nhật
        self.mis_points_ = []   # Lưu danh sách chỉ số các điểm bị phân loại sai
        self.random_state = random_state

    def _h(self, w, X):
        """Hàm dự đoán nhãn: h(w, x) = sign(w^T * x)"""
        return np.sign(np.dot(w.T, X))

    def _has_converged(self, X, y, w):
        """Kiểm tra mô hình đã dự đoán đúng 100% dữ liệu chưa"""
        return np.array_equal(self._h(w, X), y)

    def fit(self, X, y):
        """
        Huấn luyện mô hình Perceptron
        :param X: Ma trận thuộc tính có kích thước (d, N) - đã chèn hàng bias ở trên cùng
        :param y: Ma trận nhãn có kích thước (1, N) với giá trị thuộc {-1, 1}
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        d, N = X.shape
        
        # Khởi tạo trọng số ngẫu nhiên w_init có kích thước (d, 1)
        w_init = np.random.randn(d, 1)
        
        self.w_history_ = [w_init]
        self.mis_points_ = []

        while True:
            # Xáo trộn thứ tự dữ liệu ở mỗi vòng lặp
            mix_id = np.random.permutation(N)
            
            for i in range(N):
                idx = mix_id[i]
                xi = X[:, idx].reshape(d, 1)
                yi = y[0, idx]

                # Kiểm tra nếu điểm xi bị phân loại sai
                if self._h(self.w_history_[-1], xi)[0, 0] != yi:
                    self.mis_points_.append(idx)
                    # Cập nhật trọng số: w_new = w + y * x
                    w_new = self.w_history_[-1] + yi * xi
                    self.w_history_.append(w_new)

            # Kiểm tra xem toàn bộ tập dữ liệu đã hội tụ chưa
            if self._has_converged(X, y, self.w_history_[-1]):
                break

        # Lưu trọng số tối ưu cuối cùng
        self.w_ = self.w_history_[-1]
        return self

    def predict(self, X_new):
        """
        Dự báo nhãn cho dữ liệu mới
        :param X_new: Ma trận dữ liệu mới có kích thước (d, N_new) - đã bao gồm hàng bias
        :return: Ma trận nhãn dự đoán kích thước (1, N_new)
        """
        if self.w_ is None:
            raise ValueError("Mô hình chưa được huấn luyện! Vui lòng gọi hàm fit() trước.")
        
        return self._h(self.w_, X_new)

# 1. Sinh dữ liệu giả lập
np.random.seed(2)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)

# Thêm hàng bias số 1 vào ma trận dữ liệu (chuyển X thành Xbar kích thước 3x20)
Xbar = np.concatenate((np.ones((1, 2 * N)), X), axis=0)

# 2. Huấn luyện mô hình với hàm fit()
model = Perceptron(random_state=42)
model.fit(Xbar, y)

print("=== HOÀN THÀNH HUẤN LUYỆN ===")
print(f"Tổng số lần cập nhật trọng số: {len(model.w_history_) - 1}")
print("Vector trọng số tìm được (w_bar):")
print(model.w_)

# 3. Dự báo nhãn dữ liệu mới với hàm predict()
# Tạo 2 điểm dữ liệu thử nghiệm: Điểm A(2, 2) thuộc nhóm 1; Điểm B(4, 2) thuộc nhóm -1
X_test = np.array([
    [2.0, 4.0],  # Thuộc tính x1
    [2.0, 2.0]   # Thuộc tính x2
])

# Thêm hàng bias số 1 vào dữ liệu test
X_test_bar = np.concatenate((np.ones((1, 2)), X_test), axis=0)

# Dự báo
y_pred = model.predict(X_test_bar)

print("\n=== KẾT QUẢ DỰ BÁO DỮ LIỆU MỚI ===")
print("Dữ liệu test (đã chèn bias 1 ở hàng đầu):")
print(X_test_bar)
print("Nhãn dự đoán tương ứng:")
print(y_pred)