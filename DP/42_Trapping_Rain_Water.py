# Given n non-negative integers representing 
# an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6
# Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
# In this case, 6 units of rain water (blue section) are being trapped.
# Example 2:

# Input: height = [4,2,0,3,2,5]
# Output: 9

from typing import List
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right =0,  len(height) -1
        left_max, right_max = 0, 0
        res = 0

        while left < right:
            if height[left] < height[right]:
                # Xử lý left 
                if height[left] >= left_max:
                    left_max = height[left]
                else:
                    res += left_max - height[left]
                left+=1
            else:
                if height[right] >= right_max:
                    right_max = height[right]
                else:
                     res += right_max - height[right]
                right -=1
        
        return res


def main():
    height = [0,1,0,2,1,0,1,3,2,1,2,1]
    sol = Solution()
    print("Kết quả: ", sol.trap(height))

if __name__ == "__main__":
    main()



        