/**
 * LeetCode 72: Edit Distance
 * 
 * Given two strings word1 and word2, return the minimum number of operations required to 
 * convert word1 to word2.
 * 
 * You have the following three operations permitted on a word:
 * 1. Insert a character
 * 2. Delete a character
 * 3. Replace a character
 * 
 * This is the classic Levenshtein distance problem.
 */

class Solution {
    public int minDistance(String word1, String word2) {
        if (word1 == null || word2 == null) {
            return 0;
        }
        
        int m = word1.length();
        int n = word2.length();
        
        // dp[i][j] = minimum operations to convert word1[0..i-1] to word2[0..j-1]
        int[][] dp = new int[m + 1][n + 1];
        
        // Base cases
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i; // Delete all characters from word1
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j; // Insert all characters to match word2
        }
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    // Characters match, no operation needed
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    // Take minimum of three operations: replace, delete, insert
                    dp[i][j] = 1 + Math.min(
                        Math.min(dp[i - 1][j], dp[i][j - 1]),  // delete or insert
                        dp[i - 1][j - 1]                        // replace
                    );
                }
            }
        }
        
        return dp[m][n];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.minDistance("horse", "ros"));        // 3 (delete h, delete e, replace o with r)
        System.out.println(sol.minDistance("intention", "execution")); // 5
        System.out.println(sol.minDistance("a", "b"));              // 1
        System.out.println(sol.minDistance("", "abc"));             // 3
    }
}
