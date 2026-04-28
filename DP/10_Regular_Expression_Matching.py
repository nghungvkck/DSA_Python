# https://www.youtube.com/watch?v=ZNI_yXaGlxY

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        matrix = [[None] * (len(p) + 1) for _ in range(len(s) + 1)]

        def dp(i, j):
            # Kiểm tra ô hiện tại của ma trận đã được xử lí chưa
            if matrix[i][j] is not None:
                return matrix[i][j]
            
            # Kiểm tra i và j đã đến cuối mẫu chuỗi chưa
            # J đại diện cho vị trí hiện tại của chúng ta trong mẫu
            # I đại diện cho vị trí hiện tại của chúng ta trong chuỗi
            if j == len(p): 
                matrix[i][j] = (i == len(s))
                return matrix[i][j]

            if i == len(s):
                if (len(p) - j) % 2 == 1:
                    matrix[i][j] = False
                    return False
                
                for k in range(j + 1, len(p), 2):
                    if p[k] != "*":
                        matrix[i][j] = False
                        return False
                
                matrix[i][j] = True
                return True
            
            # Tiếp theo chúng ta sẽ so sánh giá trị chuỗi hiện tại
            # Với giá trị mẫu hiện tại, nếu trùng khớp, chúng ta trả về True
            match = (p[j] == s[i] or p[j] == ".")

            if j < len(p) - 1 and p[j + 1] == '*':
                result = dp(i, j + 2) or (match and dp(i + 1, j))
            else:
                result = match and dp(i + 1, j + 1)

            matrix[i][j] = result
            return result

        return dp(0, 0)