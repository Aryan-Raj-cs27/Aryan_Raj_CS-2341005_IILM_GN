/**
 * LeetCode 1047: Remove All Adjacent Duplicates In String
 * 
 * You are given a string s consisting of lowercase English letters. 
 * A duplicate removal consists of choosing two adjacent and equal letters and removing them.
 * 
 * We repeatedly make duplicate removals on s until we no longer can.
 * Return the final string after all duplicate removals have been made.
 */

import java.util.Stack;

class Solution {
    public String removeDuplicates(String s) {
        Stack<Character> stack = new Stack<>();
        
        for (char c : s.toCharArray()) {
            if (!stack.isEmpty() && stack.peek() == c) {
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        
        StringBuilder result = new StringBuilder();
        while (!stack.isEmpty()) {
            result.append(stack.pop());
        }
        
        return result.reverse().toString();
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.removeDuplicates("abbaca"));      // "ca"
        System.out.println(sol.removeDuplicates("a"));           // "a"
        System.out.println(sol.removeDuplicates("aa"));          // ""
    }
}
