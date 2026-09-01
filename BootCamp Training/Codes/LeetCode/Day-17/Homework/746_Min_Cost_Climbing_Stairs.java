/**
 * LeetCode 746: Min Cost Climbing Stairs
 * 
 * You are given an integer array cost where cost[i] is the cost of ith step on a staircase.
 * Once you pay the cost, you can climb one or two steps.
 * 
 * You can either start from the step with index 0, or the step with index 1.
 * Return the minimum cost to reach the top of the floor.
 */

class Solution {
    public int minCostClimbingStairs(int[] cost) {
        if (cost == null || cost.length == 0) {
            return 0;
        }
        if (cost.length == 1) {
            return cost[0];
        }
        
        int prev2 = cost[0], prev1 = cost[1];
        
        for (int i = 2; i < cost.length; i++) {
            int current = cost[i] + Math.min(prev1, prev2);
            prev2 = prev1;
            prev1 = current;
        }
        
        return Math.min(prev1, prev2);
    }
    
    // Dynamic Programming approach
    public int minCostClimbingStairsDP(int[] cost) {
        if (cost == null || cost.length == 0) {
            return 0;
        }
        
        int n = cost.length;
        int[] dp = new int[n];
        dp[0] = cost[0];
        dp[1] = cost[1];
        
        for (int i = 2; i < n; i++) {
            dp[i] = cost[i] + Math.min(dp[i - 1], dp[i - 2]);
        }
        
        return Math.min(dp[n - 1], dp[n - 2]);
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] cost1 = {10, 15, 20};
        System.out.println(sol.minCostClimbingStairs(cost1));  // 15 (start at index 1 and climb 2 steps)
        
        int[] cost2 = {1, 100, 1, 1, 1, 100, 1, 1, 100, 1};
        System.out.println(sol.minCostClimbingStairs(cost2));  // 6
        
        int[] cost3 = {5, 2};
        System.out.println(sol.minCostClimbingStairsDP(cost3)); // 2
    }
}
