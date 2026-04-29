# Example 1:

# Input: s = "(()"
# Output: 2
# Explanation: The longest valid parentheses substring is "()".
# Example 2:

# Input: s = ")()())"
# Output: 4
# Explanation: The longest valid parentheses substring is "()()".
# Example 3:

# Input: s = ""
# Output: 0
# trả về dấu ngoặc đơn hợp lệ dài nhất
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack =[-1]
        maxLength = 0

        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxLength = max(maxLength, i - stack[-1])
        
        return maxLength
    
def main():
    s = "(())"
    sol = Solution()
    print("Giá trị là:", sol.longestValidParentheses(s))


if __name__ == "__main__":
    main()
