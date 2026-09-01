/**
 * LeetCode 62: Unique Paths
 * 
 * There is an m x n grid. A robot is located at the top-left corner 
 * (i.e., grid[0][0]) and wants to reach the bottom-right corner (i.e., grid[m-1][n-1]).
 * 
 * The robot can only move either down or right at any point in time.
 * Given the two integers m and n, return the number of unique paths that the robot can take 
 * to reach the bottom-right corner.
 */

class Solution {
    public int uniquePaths(int m, int n) {
        // dp[i][j] = number of unique paths to reach (i, j)
        int[][] dp = new int[m][n];
        
        // Initialize first row and first column
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }
        
        // Fill the dp table
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        
        return dp[m - 1][n - 1];
    }
    
    // Space-optimized approach
    public int uniquePathsOptimized(int m, int n) {
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
        }
        
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] += dp[j - 1];
            }
        }
        
        return dp[n - 1];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.uniquePaths(3, 7));  // 28
        System.out.println(sol.uniquePaths(3, 2));  // 3
        System.out.println(sol.uniquePaths(1, 1));  // 1
        System.out.println(sol.uniquePathsOptimized(3, 7)); // 28
    }
}
