# thuật toán sinh dùng quay lui 

def get_all_subsets_backtrack(arr):
    res = []
    
    def backtrack(start, current_subset):
        res.append(list(current_subset))
        for i in range(start, len(arr)):
            current_subset.append(arr[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()  # Quay lui

    backtrack(0, [])
    return res

# Ví dụ
data = [1, 2, 3]
print(get_all_subsets_backtrack(data))