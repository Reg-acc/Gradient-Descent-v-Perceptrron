import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 1. TẠO TẬP DỮ LIỆU GIAO DỊCH GIAN LẬN (MẤT CÂN BẰNG)
# Sinh dữ liệu giả lập giống bài toán Credit Card Fraud:
# - 10,000 giao dịch với 10 đặc trưng (so sánh, số tiền, thời gian, vị trí...)
# - Lớp 0: Giao dịch hợp lệ (99%)
# - Lớp 1: Giao dịch gian lận (1%)
X, y = make_classification(
    n_samples=10000, 
    n_features=10, 
    n_informative=8, 
    n_redundant=2,
    weights=[0.99, 0.01], 
    random_state=42
)

# Chia tập huấn luyện (80%) và tập kiểm thử (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. TIỀN XỬ LÝ DỮ LIỆU & XỬ LÝ MẤT CÂN BẰNG
# Bước 1: Chuẩn hóa dữ liệu (Standardization) - Bắt buộc với Perceptron
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bước 2: Cân bằng dữ liệu bằng SMOTE (Chỉ áp dụng trên tập Train)
# Kỹ thuật này giúp Perceptron "nhận biết" được các mẫu gian lận hiếm hoi
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# 3. HUẤN LUYỆN MÔ HÌNH PERCEPTRON
model = Perceptron(
    max_iter=1000, 
    eta0=0.01, 
    random_state=42, 
    class_weight='balanced'
)

# Huấn luyện trên tập dữ liệu đã qua xử lý
model.fit(X_train_resampled, y_train_resampled)

# 4. DỰ BÁO VÀ ĐÁNH GIÁ ĐỘ ĐO
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH PERCEPTRON ===")
print(f"1. Accuracy  (Độ chính xác toàn cục) : {acc * 100:.2f}%")
print(f"2. Precision (Độ chính xác gian lận) : {prec * 100:.2f}%")
print(f"3. Recall    (Độ nhạy bắt gian lận)  : {rec * 100:.2f}%")
print(f"4. F1-score  (Cân bằng Prec/Rec)     : {f1 * 100:.2f}%")

print("\n=== MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== BÁO CÁO CHI TIẾT (CLASSIFICATION REPORT) ===")
print(classification_report(y_test, y_pred, target_names=['Hợp lệ (0)', 'Gian lận (1)']))