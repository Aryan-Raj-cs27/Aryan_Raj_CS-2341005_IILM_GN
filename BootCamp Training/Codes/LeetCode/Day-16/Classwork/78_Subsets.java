/**
 * LeetCode 78: Subsets
 * 
 * Given an integer array nums of unique elements, return all possible subsets (the power set).
 * The solution set must not contain duplicate subsets.
 * You may return the solution in any order.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> current, int[] nums, int start) {
        result.add(new ArrayList<>(current));
        
        for (int i = start; i < nums.length; i++) {
            current.add(nums[i]);
            backtrack(result, current, nums, i + 1);
            current.remove(current.size() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 2, 3};
        System.out.println(sol.subsets(nums1));
        // [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
        
        int[] nums2 = {0};
        System.out.println(sol.subsets(nums2));
        // [[], [0]]
    }
}
