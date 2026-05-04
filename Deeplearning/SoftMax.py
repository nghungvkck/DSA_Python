import math

def softmax(z):
    """
    z: list hoặc list of lists (mỗi hàng là một mẫu, mỗi cột là điểm số của một lớp)
    Trả về: xác suất sau softmax (cùng shape)
    """
    # Nếu z là vector 1D
    if isinstance(z[0], (int, float)):
        max_z = max(z)
        exp_z = [math.exp(zi - max_z) for zi in z]  # trừ max để tránh tràn số
        sum_exp = sum(exp_z)
        return [e / sum_exp for e in exp_z]
    # Nếu z là ma trận 2D
    else:
        result = []
        for row in z:
            max_row = max(row)
            exp_row = [math.exp(x - max_row) for x in row]
            sum_exp = sum(exp_row)
            result.append([e / sum_exp for e in exp_row])
        return result