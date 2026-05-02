def sinh_chinh_hop(arr, k):
    res = []
    n = len(arr)
    used = [False] * n # Mảng đánh dấu

    def backtrack(path):
        if len(path) == k:
            res.append(list(path))
            return
        
        for i in range(n):
            if not used[i]:
                used[i] = True       # Đánh dấu đã dùng
                path.append(arr[i])
                
                backtrack(path)      # Đệ quy
                
                path.pop()           # Quay lui
                used[i] = False      # Bỏ đánh dấu

    backtrack([])
    return res

# Chạy thử
data = [1, 2, 3]
k = 3
result = sinh_chinh_hop(data, k)
print(f"Số lượng: {len(result)}")
for item in result:
    print(item)