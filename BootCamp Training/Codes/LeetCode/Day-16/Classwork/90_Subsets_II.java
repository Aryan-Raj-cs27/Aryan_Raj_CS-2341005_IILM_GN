/**
 * LeetCode 90: Subsets II
 * 
 * Given an integer array nums that may contain duplicates, return all possible subsets 
 * (the power set).
 * 
 * The solution set must not contain duplicate subsets. Return the solution in any order.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(result, new ArrayList<>(), nums, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> current, int[] nums, int start) {
        result.add(new ArrayList<>(current));
        
        for (int i = start; i < nums.length; i++) {
            // Skip duplicates
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            
            current.add(nums[i]);
            backtrack(result, current, nums, i + 1);
            current.remove(current.size() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 2, 2};
        System.out.println(sol.subsetsWithDup(nums1));
        // [[], [1], [1,2], [1,2,2], [2], [2,2]]
        
        int[] nums2 = {4, 4, 4, 1, 0};
        System.out.println(sol.subsetsWithDup(nums2));
    }
}
