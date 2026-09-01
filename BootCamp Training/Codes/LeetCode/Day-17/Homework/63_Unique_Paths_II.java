/**
 * LeetCode 63: Unique Paths II
 * 
 * You are given an m x n integer array grid. There is a robot located at the top-left corner 
 * (grid[0][0]) and it wants to reach the bottom-right corner (grid[m-1][n-1]).
 * 
 * The robot can only move either down or right at any point in time.
 * An obstacle and space are marked as 1 or 0 respectively in grid.
 * A path that the robot takes cannot include any cell that is an obstacle.
 * 
 * Return the number of unique paths that the robot can take to reach the bottom-right corner.
 */

class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        if (obstacleGrid[0][0] == 1) {
            return 0;
        }
        
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        
        // Initialize first cell
        dp[0][0] = 1;
        
        // Initialize first row
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[0][j] == 0) {
                dp[0][j] = dp[0][j - 1];
            }
        }
        
        // Initialize first column
        for (int i = 1; i < m; i++) {
            if (obstacleGrid[i][0] == 0) {
                dp[i][0] = dp[i - 1][0];
            }
        }
        
        // Fill the dp table
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        
        return dp[m - 1][n - 1];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[][] grid1 = {{0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
        System.out.println(sol.uniquePathsWithObstacles(grid1));  // 2
        
        int[][] grid2 = {{0, 1}, {0, 0}};
        System.out.println(sol.uniquePathsWithObstacles(grid2));  // 1
        
        int[][] grid3 = {{0}};
        System.out.println(sol.uniquePathsWithObstacles(grid3));  // 1
        
        int[][] grid4 = {{1}};
        System.out.println(sol.uniquePathsWithObstacles(grid4));  // 0
    }
}
