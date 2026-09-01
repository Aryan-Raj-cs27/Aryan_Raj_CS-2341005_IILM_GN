/**
 * LeetCode 213: House Robber II
 * 
 * You are a professional robber planning to rob houses along a street. 
 * Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. 
 * That means the first house is the neighbor of the last one.
 * 
 * You cannot rob two adjacent houses. You have a constraint that you cannot rob the first and 
 * last house at the same time.
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
        if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }
        
        // Case 1: Rob houses from index 0 to n-2 (exclude last house)
        int max1 = robLinear(nums, 0, nums.length - 2);
        
        // Case 2: Rob houses from index 1 to n-1 (exclude first house)
        int max2 = robLinear(nums, 1, nums.length - 1);
        
        return Math.max(max1, max2);
    }
    
    private int robLinear(int[] nums, int start, int end) {
        int prev2 = 0, prev1 = 0;
        
        for (int i = start; i <= end; i++) {
            int current = Math.max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = current;
        }
        
        return prev1;
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 2, 3, 1};
        System.out.println(sol.rob(nums1));  // 4 (rob house 1 and 3)
        
        int[] nums2 = {2, 3, 2};
        System.out.println(sol.rob(nums2));  // 3 (rob house 2)
        
        int[] nums3 = {1, 2, 1, 1, 4};
        System.out.println(sol.rob(nums3));  // 6 (rob houses 1, 3, and 5)
    }
}
