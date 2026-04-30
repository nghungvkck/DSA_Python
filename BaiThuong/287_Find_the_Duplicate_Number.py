# Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

# There is only one repeated number in nums, return this repeated number.

# You must solve the problem without modifying the array nums and using only constant extra space.

 

# Example 1:

# Input: nums = [1,3,4,2,2]
# Output: 2
# Example 2:

# Input: nums = [3,1,3,4,2]
# Output: 3
# Example 3:

# Input: nums = [3,3,3,3,3]
# Output: 3
 
from typing import List
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # seen = set()

        # for x in nums:
        #     if x in seen:
        #         return x
        #     seen.add(x)


        arr = []
        left = 0
        right = len(nums) - 1

        while left < right:
            if not arr:
                arr.append(nums[left])
                if nums[right] in arr:
                    return nums[right]
                left+=1
            else:
                if nums[left] in arr:
                    return nums[left]
                else:
                    arr.append(nums[left])

                if nums[right] in arr:
                    return nums[right]
                else:
                    arr.append(nums[right])

                left += 1
                right -= 1
        return 0
    
def main():
    nums =[1,1]
    sol = Solution()
    print("result ", sol.findDuplicate(nums))

if __name__ == "__main__":
    main()