# Ví dụ 1:

# Đầu vào: prices = [3,3,5,0,0,3,1,4]
#  Đầu ra: 6
#  Giải thích: Mua vào ngày 4 (giá = 0) và bán vào ngày 6 (giá = 3), lợi nhuận = 3-0 = 3.
# Sau đó mua vào ngày thứ 7 (giá = 1) và bán vào ngày thứ 8 (giá = 4), lợi nhuận = 4-1 = 3.
# Ví dụ 2:

# Đầu vào: prices = [1,2,3,4,5]
#  Đầu ra: 4
#  Giải thích: Mua vào ngày 1 (giá = 1) và bán vào ngày 5 (giá = 5), lợi nhuận = 5-1 = 4.
# Lưu ý rằng bạn không thể mua vào ngày 1, mua vào ngày 2 và bán chúng sau đó, vì bạn đang thực hiện nhiều giao dịch cùng một lúc. Bạn phải bán trước khi mua lại.
# Ví dụ 3:

# Đầu vào: prices = [7,6,4,3,1]
#  Đầu ra: 0
#  Giải thích: Trong trường hợp này, không có giao dịch nào được thực hiện, tức là lợi nhuận tối đa = 0.


from typing import List
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # ý tưởng là chia mảng thành 2 phần
        # b1 là duyệt từ trái qua phải
        # left[i] = lợi nhuận tốt nhất từ [0 → i]
        # b2 là duyệt từ phải qua trái
        # right[i] = lợi nhuận tốt nhất từ [i → n-1]
        # b3: ghép lại

        n = len(prices)
        left = [0] * n
        right = [0] * n 

        # Duyệt từ trái qua phải, tìm lợi nhất lớn nhất lần 1 từ 0 đến i 
        min_price = prices[0]
        for i in range(1, n):
            min_price = min(min_price, prices[i])
            left[i] = max(left[i-1], prices[i] - min_price)
        
        print("left: ", left)

        # Duyệt từ phải qua trai, tìm lọi nhuận lớn nất từ i đến n
        max_price = prices[n-1]
        for i in range(n-2, -1, -1):
             max_price = max(max_price, prices[i])
             right[i] = max(right[i+1], max_price - prices[i])

        print("right: ", right)
        
        # combine
        res = 0
        for i in range(n):
             res = max(res,  left[i]+ right[i])
        return res


def main():
        prices = [3, 3, 5, 0, 0, 3, 1, 4]
        sol = Solution()
        result = sol.maxProfit(prices)
        print("Gia tri lon nhat la: ", result)

if __name__ == "__main__":
     main()
