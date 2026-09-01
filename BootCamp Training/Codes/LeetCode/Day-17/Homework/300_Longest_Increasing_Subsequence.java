/**
 * LeetCode 300: Longest Increasing Subsequence
 * 
 * Given an integer array nums, return the length of the longest strictly increasing subsequence.
 * 
 * A subsequence is a sequence that can be derived from an array by deleting some or no elements 
 * without changing the order of the remaining elements.
 */

import java.util.Arrays;

class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        
        int n = nums.length;
        
        // dp[i] = length of longest increasing subsequence ending at index i
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }
        
        int max = 0;
        for (int length : dp) {
            max = Math.max(max, length);
        }
        
        return max;
    }
    
    // Binary search approach - O(n log n)
    public int lengthOfLISBinarySearch(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        
        int[] tails = new int[nums.length];
        int len = 0;
        
        for (int num : nums) {
            int pos = Arrays.binarySearch(tails, 0, len, num);
            if (pos < 0) {
                pos = -(pos + 1);
            }
            tails[pos] = num;
            if (pos == len) {
                len++;
            }
        }
        
        return len;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {10, 9, 2, 5, 3, 7, 101, 18};
        System.out.println(sol.lengthOfLIS(nums1));  // 4 (2, 3, 7, 101)
        
        int[] nums2 = {0, 1, 0, 4, 4, 4, 3, 2, 1};
        System.out.println(sol.lengthOfLIS(nums2));  // 2 (0, 1) or (0, 4)
        
        int[] nums3 = {1};
        System.out.println(sol.lengthOfLISBinarySearch(nums3));  // 1
        
        int[] nums4 = {3, 10, 2, 1, 20};
        System.out.println(sol.lengthOfLISBinarySearch(nums4));  // 3 (3, 10, 20)
    }
}
