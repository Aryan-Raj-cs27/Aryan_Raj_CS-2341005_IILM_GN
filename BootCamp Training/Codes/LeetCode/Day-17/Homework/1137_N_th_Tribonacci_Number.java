/**
 * LeetCode 1137: N-th Tribonacci Number
 * 
 * The Tribonacci sequence Tn is defined as follows:
 * T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.
 * 
 * Given n, return the value of Tn.
 */

class Solution {
    public int tribonacci(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        
        int a = 0, b = 1, c = 1;
        for (int i = 3; i <= n; i++) {
            int next = a + b + c;
            a = b;
            b = c;
            c = next;
        }
        
        return c;
    }
    
    // Dynamic Programming approach
    public int tribonacciDP(int n) {
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 1;
        
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3];
        }
        
        return dp[n];
    }
    
    // Memoization approach
    public int tribonacciMemo(int n) {
        int[] memo = new int[n + 1];
        return tribonacciHelper(n, memo);
    }
    
    private int tribonacciHelper(int n, int[] memo) {
        if (n == 0) {
            return 0;
        }
        if (n == 1 || n == 2) {
            return 1;
        }
        if (memo[n] != 0) {
            return memo[n];
        }
        
        memo[n] = tribonacciHelper(n - 1, memo) + 
                  tribonacciHelper(n - 2, memo) + 
                  tribonacciHelper(n - 3, memo);
        return memo[n];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.tribonacci(0));   // 0
        System.out.println(sol.tribonacci(1));   // 1
        System.out.println(sol.tribonacci(4));   // 2 (0, 1, 1, 2, 4)
        System.out.println(sol.tribonacci(10));  // 81
        System.out.println(sol.tribonacciDP(25)); // 1389537
    }
}
