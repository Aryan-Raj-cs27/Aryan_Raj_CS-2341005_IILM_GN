/**
 * LeetCode 198: House Robber
 * 
 * You are a professional robber planning to rob houses along a street. 
 * Each house has a certain amount of money stashed.
 * 
 * Constraint: You cannot rob two adjacent houses because connected security systems will 
 * automatically contact the police if two adjacent houses were both broken into on the same night.
 * 
 * Given an integer array nums representing the amount of money in each house, 
 * return the maximum amount of money you can rob.
 */

class Solution {
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        
        int prev = 0, curr = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            int next = Math.max(curr, prev + nums[i]);
            prev = curr;
            curr = next;
        }
        
        return curr;
    }
    
    // Dynamic Programming approach
    public int robDP(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        
        return dp[n - 1];
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 2, 3, 1};
        System.out.println(sol.rob(nums1));  // 4 (rob house 0 and 2: 1 + 3 = 4)
        
        int[] nums2 = {2, 7, 9, 3};
        System.out.println(sol.rob(nums2));  // 9 (rob house 1 and 3: 7 + 3 = 10, or just house 2: 9)
        
        int[] nums3 = {5, 3, 4, 11, 2};
        System.out.println(sol.robDP(nums3)); // 16 (rob houses 0, 2, 4: 5 + 4 + 2 = 11, or 0, 3: 5 + 11 = 16)
    }
}
