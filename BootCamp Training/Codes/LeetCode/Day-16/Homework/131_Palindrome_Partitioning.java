/**
 * LeetCode 131: Palindrome Partitioning
 * 
 * Given a string s, partition s such that every substring of the partition is a palindrome.
 * Return all possible palindrome partitioning of s.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), s, 0);
        return result;
    }
    
    private void backtrack(List<List<String>> result, List<String> current, String s, int start) {
        if (start == s.length()) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        for (int i = start; i < s.length(); i++) {
            if (isPalindrome(s, start, i)) {
                current.add(s.substring(start, i + 1));
                backtrack(result, current, s, i + 1);
                current.remove(current.size() - 1);
            }
        }
    }
    
    private boolean isPalindrome(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.partition("nitin"));
        // [["n","i","t","i","n"], ["n","iti","n"]]
        
        System.out.println(sol.partition("a"));
        // [["a"]]
    }
}
