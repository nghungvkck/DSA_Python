# You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and the height of an envelope.

# One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.

# Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).

# Note: You cannot rotate an envelope.

 

# Example 1:

# Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
# Output: 3
# Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
# Example 2:

# Input: envelopes = [[1,1],[1,1],[1,1]]
# Output: 1
from typing import List
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        envelopes.sort(key=lambda x: (x[0] , x[1]))

        n = len(envelopes)
        dp =[1] * n

        for i in range(n):
            for j in range(i):
                if (envelopes[j][0] < envelopes[i][0] and 
                    envelopes[j][1] < envelopes[i][1]):
                    # dp[j] +1, só lồng tốt nhát đến j
                    # dp[i], số phòng bì tốt nhất có thể lồng được 
                    # và kết thúc tại i
                    dp[i] = max(dp[i], dp[j] +1)
        
        return max(dp)
        
    

def main():
    envelopes = [[5,4],[6,4],[6,7],[2,3]]
    sol = Solution()
    print("Giá trị hợp lệ là ", sol.maxEnvelopes(envelopes=envelopes))

if __name__ == "__main__":
    main()