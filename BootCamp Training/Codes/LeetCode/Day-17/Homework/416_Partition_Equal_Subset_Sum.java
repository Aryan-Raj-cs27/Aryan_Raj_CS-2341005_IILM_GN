/**
 * LeetCode 416: Partition Equal Subset Sum
 * 
 * Given an integer array nums, return true if you can partition the array into two subsets 
 * such that the sum of elements in both subsets is equal or false otherwise.
 */

class Solution {
    public boolean canPartition(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        
        // If sum is odd, cannot partition into equal subsets
        if (sum % 2 != 0) {
            return false;
        }
        
        int target = sum / 2;
        
        // dp[i] = true if we can achieve sum i using the array elements
        boolean[] dp = new boolean[target + 1];
        dp[0] = true; // We can always achieve sum 0 by selecting no elements
        
        for (int num : nums) {
            // Traverse from right to left to avoid using the same element twice
            for (int i = target; i >= num; i--) {
                dp[i] = dp[i] || dp[i - num];
            }
        }
        
        return dp[target];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 5, 11, 5};
        System.out.println(sol.canPartition(nums1));  // true (11 = 5 + 5 + 1)
        
        int[] nums2 = {2, 2, 1, 1};
        System.out.println(sol.canPartition(nums2));  // true (2 + 1 = 2 + 1)
        
        int[] nums3 = {1, 2, 5};
        System.out.println(sol.canPartition(nums3));  // false
        
        int[] nums4 = {1};
        System.out.println(sol.canPartition(nums4));  // false
    }
}
