# Cho một mảng numschứa ncác đối tượng có màu đỏ, trắng hoặc xanh lam, 
# hãy sắp xếp chúng tại chỗ sao cho các đối tượng cùng màu nằm cạnh nhau, 
# theo thứ tự màu sắc là đỏ, trắng và xanh lam.

# Chúng ta sẽ sử dụng các số nguyên 0, 1, và 2để lần lượt biểu thị màu đỏ, trắng và xanh lam.

# Bạn phải giải quyết vấn đề này mà không sử dụng hàm sắp xếp của thư viện.

 

# Ví dụ 1:

# Đầu vào: nums = [2,0,2,1,1,0]
#  Đầu ra: [0,0,1,1,2,2]
# Ví dụ 2:

# Đầu vào: nums = [2,0,1]
#  Đầu ra: [0,1,2]
 

from typing import List
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        # """
        # Do not return anything, modify nums in-place instead.
        # """
        for i in range(len(nums)):
            for j in range(0, len(nums)-i-1):
                if nums[j] > nums[j+1]:  # đổi dấu ở đây
                    temp = nums[j]
                    nums[j] = nums[j+1]
                    nums[j+1] = temp

        return self.quickSort(nums)

    def quickSort(self, nums):
        if len(nums) <= 1:
            return nums

        pivot = nums[len(nums)//2]
        left = []
        middle = []
        right = []

        for x in nums:
            if x < pivot:
                left.append(x)
            elif x > pivot:
                right.append(x)
            else:
                middle.append(x)

        return self.quickSort(left) + middle + self.quickSort(right)


def main():
    nums = [2,0,2,1,1,0]
    sol = Solution()
    print("kết quả là: ", sol.sortColors(nums=nums))

if __name__ == "__main__":
    main()
        