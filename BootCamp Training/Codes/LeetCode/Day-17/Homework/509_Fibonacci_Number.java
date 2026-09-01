/**
 * LeetCode 509: Fibonacci Number
 * 
 * The Fibonacci numbers, commonly denoted F(n) form a sequence, called the Fibonacci sequence, 
 * such that each number is the sum of the two preceding ones, starting from 0 and 1.
 * 
 * Given n, calculate F(n).
 */

class Solution {
    public int fib(int n) {
        if (n <= 1) {
            return n;
        }
        
        int prev = 0, curr = 1;
        for (int i = 2; i <= n; i++) {
            int next = prev + curr;
            prev = curr;
            curr = next;
        }
        
        return curr;
    }
    
    // Recursive approach with memoization (alternative)
    public int fibMemo(int n) {
        int[] memo = new int[n + 1];
        return fibHelper(n, memo);
    }
    
    private int fibHelper(int n, int[] memo) {
        if (n <= 1) {
            return n;
        }
        if (memo[n] != 0) {
            return memo[n];
        }
        
        memo[n] = fibHelper(n - 1, memo) + fibHelper(n - 2, memo);
        return memo[n];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.fib(0));   // 0
        System.out.println(sol.fib(1));   // 1
        System.out.println(sol.fib(4));   // 3 (0, 1, 1, 2, 3)
        System.out.println(sol.fib(10));  // 55
        System.out.println(sol.fibMemo(10)); // 55
    }
}
