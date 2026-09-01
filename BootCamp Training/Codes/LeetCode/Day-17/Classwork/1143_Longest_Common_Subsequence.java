/**
 * LeetCode 1143: Longest Common Subsequence
 * 
 * Given two strings text1 and text2, return the length of their longest common subsequence. 
 * If there is no common subsequence, return 0.
 * 
 * A subsequence of a string is a new string generated from the original string with some 
 * characters (can be none) deleted without changing the relative order of the remaining characters.
 */

class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        if (text1 == null || text2 == null || text1.length() == 0 || text2.length() == 0) {
            return 0;
        }
        
        int m = text1.length();
        int n = text2.length();
        
        // dp[i][j] = length of LCS of text1[0..i-1] and text2[0..j-1]
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    // Characters match, extend the LCS
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    // Characters don't match, take the maximum from excluding one of them
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.longestCommonSubsequence("abcde", "ace"));      // 3 (ace)
        System.out.println(sol.longestCommonSubsequence("abc", "abc"));        // 3
        System.out.println(sol.longestCommonSubsequence("abc", "def"));        // 0
        System.out.println(sol.longestCommonSubsequence("intention", "execution")); // 5
    }
}
