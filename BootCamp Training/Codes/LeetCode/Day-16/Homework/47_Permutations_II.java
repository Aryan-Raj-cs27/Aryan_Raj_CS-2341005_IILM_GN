/**
 * LeetCode 47: Permutations II
 * 
 * Given a collection of numbers, nums, that may contain duplicates, 
 * return all possible permutations in any order.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(result, new ArrayList<>(), nums, new boolean[nums.length]);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> current, int[] nums, boolean[] used) {
        if (current.size() == nums.length) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        for (int i = 0; i < nums.length; i++) {
            // Skip duplicates and already used numbers
            if (used[i] || (i > 0 && nums[i] == nums[i - 1] && !used[i - 1])) {
                continue;
            }
            
            used[i] = true;
            current.add(nums[i]);
            backtrack(result, current, nums, used);
            current.remove(current.size() - 1);
            used[i] = false;
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] nums1 = {1, 1, 2};
        System.out.println(sol.permuteUnique(nums1));
        // [[1,1,2], [1,2,1], [2,1,1]]
        
        int[] nums2 = {0, 1};
        System.out.println(sol.permuteUnique(nums2));
        // [[0,1], [1,0]]
    }
}
