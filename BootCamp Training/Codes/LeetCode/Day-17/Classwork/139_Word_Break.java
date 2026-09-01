/**
 * LeetCode 139: Word Break
 * 
 * Given a string s and a dictionary of strings wordDict, 
 * return true if s can be segmented into a space-separated sequence of dictionary words.
 * 
 * You may assume the dictionary does not contain duplicate words.
 */

import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.length() == 0 || wordDict == null || wordDict.isEmpty()) {
            return s.length() == 0;
        }
        
        Set<String> words = new HashSet<>(wordDict);
        int n = s.length();
        
        // dp[i] = true if s[0..i-1] can be segmented into words from the dictionary
        boolean[] dp = new boolean[n + 1];
        dp[0] = true; // Base case: empty string can be segmented
        
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                // If s[0..j-1] can be segmented and s[j..i-1] is in dictionary
                if (dp[j] && words.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[n];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        List<String> dict1 = List.of("leet", "code");
        System.out.println(sol.wordBreak("leetcode", dict1));  // true
        
        List<String> dict2 = List.of("applepenapple", "apple", "pen");
        System.out.println(sol.wordBreak("applepenapple", dict2)); // true
        
        List<String> dict3 = List.of("cats", "cat", "and", "sand", "dog");
        System.out.println(sol.wordBreak("catsandog", dict3));    // false
        
        List<String> dict4 = List.of("a");
        System.out.println(sol.wordBreak("a", dict4));            // true
    }
}
