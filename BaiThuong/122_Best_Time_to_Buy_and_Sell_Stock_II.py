
# Ví dụ 1:

# Đầu vào: prices = [7,1,5,3,6,4]
#  Đầu ra: 7
#  Giải thích: Mua vào ngày 2 (giá = 1) và bán vào ngày 3 (giá = 5), lợi nhuận = 5-1 = 4.
# Sau đó mua vào ngày thứ 4 (giá = 3) và bán vào ngày thứ 5 (giá = 6), lợi nhuận = 6-3 = 3.
# Tổng lợi nhuận là 4 + 3 = 7.
# Ví dụ 2:

# Đầu vào: prices = [1,2,3,4,5]
#  Đầu ra: 4
#  Giải thích: Mua vào ngày 1 (giá = 1) và bán vào ngày 5 (giá = 5), lợi nhuận = 5-1 = 4.
# Tổng lợi nhuận là 4.
# Ví dụ 3:

# Đầu vào: prices = [7,6,4,3,1]
#  Đầu ra: 0
#  Giải thích: Không có cách nào để có được lợi nhuận dương, vì vậy chúng ta không bao giờ mua cổ phiếu để đạt được lợi nhuận tối đa là 0.

from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # hướng giải là cứ ngày nay giá cao hơn ngày trước thì ta ăn lợi nhuận 
        total = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                total += prices[i] - prices[i - 1]
        return total
