/**
 * LeetCode 518: Coin Change II
 * 
 * You are given an integer array coins representing coins of different denominations 
 * and an integer amount representing a total amount of money.
 * 
 * Return the number of combinations that make up that amount. 
 * If that amount of money cannot be made up by any combination of the coins, return 0.
 * 
 * You may assume that you have an infinite number of each kind of coin.
 * The answer is guaranteed to fit into a signed 32-bit integer.
 */

class Solution {
    public int change(int amount, int[] coins) {
        if (amount == 0) {
            return 1;
        }
        if (coins == null || coins.length == 0) {
            return 0;
        }
        
        // dp[i] = number of combinations to make up amount i
        int[] dp = new int[amount + 1];
        dp[0] = 1; // Base case: one way to make amount 0 (use no coins)
        
        // For each coin
        for (int coin : coins) {
            // Update all amounts that can be formed with this coin
            for (int i = coin; i <= amount; i++) {
                dp[i] += dp[i - coin];
            }
        }
        
        return dp[amount];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] coins1 = {1, 2, 5};
        System.out.println(sol.change(5, coins1));  // 4 (5=5, 5=2+2+1, 5=2+1+1+1, 5=1+1+1+1+1)
        
        int[] coins2 = {2};
        System.out.println(sol.change(5, coins2));  // 0
        
        int[] coins3 = {10};
        System.out.println(sol.change(10, coins3)); // 1
        
        int[] coins4 = {1, 2, 5};
        System.out.println(sol.change(3, coins4));  // 2 (3=1+1+1, 3=1+2)
    }
}
