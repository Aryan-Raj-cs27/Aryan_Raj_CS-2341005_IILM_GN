/**
 * LeetCode 46: Permutations
 * 
 * Given an array nums of distinct integers, return all the possible permutations.
 * You may return the answer in any order.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), nums, new boolean[nums.length]);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> current, int[] nums, boolean[] used) {
        if (current.size() == nums.length) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (!used[i]) {
                used[i] = true;
                current.add(nums[i]);
                backtrack(result, current, nums, used);
                current.remove(current.size() - 1);
                used[i] = false;
            }
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 2, 3};
        System.out.println(sol.permute(nums1));
        // [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
        
        int[] nums2 = {0, 1};
        System.out.println(sol.permute(nums2));
        // [[0,1], [1,0]]
        
        int[] nums3 = {1};
        System.out.println(sol.permute(nums3));
        // [[1]]
    }
}
