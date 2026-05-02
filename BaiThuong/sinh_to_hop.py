def sinh_to_hop_quay_lui(arr, k):
    res = []
    n = len(arr)

    def backtrack(start, path):
        # Nếu độ dài path bằng k, đã tìm xong 1 tổ hợp
        if len(path) == k:
            res.append(list(path))
            return
        
        # Duyệt qua các phần tử từ vị trí start
        for i in range(start, n):
            path.append(arr[i])
            backtrack(i + 1, path) # Đệ quy cho vị trí tiếp theo
            path.pop()             # Quay lui: bỏ phần tử vừa thêm để thử cái khác

    backtrack(0, [])
    return res

# Chạy thử
data = [1, 2, 3, 4]
k = 3
print(sinh_to_hop_quay_lui(data, k))