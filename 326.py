from __future__ import division, print_function, unicode_literals
import numpy as np

# 1. Định nghĩa đạo hàm và hàm chi phí (cost function)
def grad(x):
    return 2 * x - 4

def cost(x):
    return x**2 - 4*x + 5

# Thuật toán GD thực hiện 4 bước cập nhật
def myGD_4steps(eta, x0):
    x = [x0]
    for it in range(4):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)
    return x

# Thực hiện với x0 = 5, eta = 0.2
eta = 0.2
x0 = 5
x_history = myGD_4steps(eta, x0)

# In kết quả từng bước
print("Lịch sử cập nhật qua 4 bước:")
for i, x_val in enumerate(x_history):
    print("Bước %d: x = %.6f, f(x) = %.6f, f'(x) = %.6f" % (i, x_val, cost(x_val), grad(x_val)))