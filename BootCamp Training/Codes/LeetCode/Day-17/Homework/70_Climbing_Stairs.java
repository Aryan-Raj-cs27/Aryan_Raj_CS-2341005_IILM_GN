/**
 * LeetCode 70: Climbing Stairs
 * 
 * You are climbing a staircase. It takes n steps to reach the top.
 * Each time you can climb 1 or 2 steps. In how many distinct ways can you climb to the top?
 */

class Solution {
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        
        int prev = 1, curr = 2;
        for (int i = 3; i <= n; i++) {
            int next = prev + curr;
            prev = curr;
            curr = next;
        }
        
        return curr;
    }
    
    // Dynamic Programming approach
    public int climbStairsDP(int n) {
        if (n <= 2) {
            return n;
        }
        
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        return dp[n];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.climbStairs(1));  // 1
        System.out.println(sol.climbStairs(2));  // 2
        System.out.println(sol.climbStairs(3));  // 3 (1+1+1, 1+2, 2+1)
        System.out.println(sol.climbStairs(4));  // 5
        System.out.println(sol.climbStairsDP(5)); // 8
    }
}
