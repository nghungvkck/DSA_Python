# Given a triangle array, return the minimum path sum from top to bottom.

# For each step, you may move to an adjacent number of the row below. 
# More formally, if you are on index i on the current row, you may move to either index i or index i + 1 
# on the next row.


# Example 1:
# Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
# Output: 11
# Explanation: The triangle looks like:
#    2
#   3 4
#  6 5 7
# 4 1 8 3
# The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).
# Example 2:

# Input: triangle = [[-10]]
# Output: -10

# đi từ trên xuống dưới
# mỗi bước chỉ đi:

# xuống trái (i+1; j)
# xuống phải (i+1; j+1)

# tìm tổng nhỏ nhất


from typing import List
class Solution:
    # def minimumTotal(self, triangle: List[List[int]]) -> int:
    #     sum = 0
    #     for i in range(len(triangle)):
    #         sum += min(triangle[i])
    #     return sum

    def minimumTotal(self, triangle: List[List[int]]) -> int:
        # Copy hang cuoi cung 
        dp = triangle[-1][:] 

        for i in range(len(triangle)-2, -1, -1):
            for j in range(len(triangle[i])):
                dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
        return dp[0]

def main():
    triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
    sol = Solution()
    print("Kết quả ", sol.minimumTotal(triangle))

if __name__ == "__main__":
    main()