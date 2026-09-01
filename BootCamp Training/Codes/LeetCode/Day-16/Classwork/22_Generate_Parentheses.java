/**
 * LeetCode 22: Generate Parentheses
 * 
 * Given n pairs of parentheses, write a function to generate all combinations 
 * of well-formed parentheses.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        backtrack(result, new StringBuilder(), 0, 0, n);
        return result;
    }
    
    private void backtrack(List<String> result, StringBuilder current, int open, int close, int n) {
        if (current.length() == 2 * n) {
            result.add(current.toString());
            return;
        }
        
        // Add opening parenthesis if we haven't used all n opening parentheses
        if (open < n) {
            current.append('(');
            backtrack(result, current, open + 1, close, n);
            current.deleteCharAt(current.length() - 1);
        }
        
        // Add closing parenthesis if it doesn't exceed opening parentheses
        if (close < open) {
            current.append(')');
            backtrack(result, current, open, close + 1, n);
            current.deleteCharAt(current.length() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.generateParenthesis(3));
        // ["((()))", "(()())", "(())()", "()(())", "()()()"]
        
        System.out.println(sol.generateParenthesis(1));
        // ["()"]
    }
}
