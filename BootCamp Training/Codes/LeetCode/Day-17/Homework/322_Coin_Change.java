/**
 * LeetCode 322: Coin Change
 * 
 * You are given an integer array coins representing coins of different denominations 
 * and an integer amount representing a total amount of money.
 * 
 * Return the fewest number of coins that you need to make up that amount. 
 * If that amount of money cannot be made up by any combination of the coins, return -1.
 * 
 * You may assume that you have an infinite number of each kind of coin.
 */

import java.util.Arrays;

class Solution {
    public int coinChange(int[] coins, int amount) {
        if (amount == 0) {
            return 0;
        }
        if (coins == null || coins.length == 0) {
            return -1;
        }
        
        // dp[i] = minimum coins needed to make amount i
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1); // Initialize with max value
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        
        return dp[amount] > amount ? -1 : dp[amount];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] coins1 = {1, 2, 5};
        System.out.println(sol.coinChange(coins1, 5));   // 1 (5)
        
        int[] coins2 = {2};
        System.out.println(sol.coinChange(coins2, 3));   // -1
        
        int[] coins3 = {10};
        System.out.println(sol.coinChange(coins3, 10));  // 1
        
        int[] coins4 = {1, 2, 5};
        System.out.println(sol.coinChange(coins4, 7));   // 2 (5 + 2)
    }
}
