# Example 1:

# Input: s = "leetcode", wordDict = ["leet","code"]
# Output: true
# Explanation: Return true because "leetcode" can be segmented as "leet code".
# Example 2:

# Input: s = "applepenapple", wordDict = ["apple","pen"]
# Output: true
# Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
# Note that you are allowed to reuse a dictionary word.
# Example 3:

# Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
# Output: false

from typing import List
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # Nếu ko chứa từ trong từ điển thì trả về false
        # length = 0
        # for i in range(len(wordDict)):
        #     if not s.__contains__(wordDict[i]):
        #         return False
        #     else:
        #         length += s.count(wordDict[i])*len(wordDict[i])


        # if length == len(s):
        #     return True     
        # else:
        #     return False
        n = len(s)
        dp =[False] * (n+1)
        dp[0] = True

        for i in range(1, n+1):
            for j in range(i):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        
        return dp[n]


    
def main():
    s = "leetcode"
    wordDict = ["leet","code"]
    sol = Solution()
    print("Kết quả: ", sol.wordBreak(s,wordDict=wordDict))

if __name__ == "__main__":
    main()