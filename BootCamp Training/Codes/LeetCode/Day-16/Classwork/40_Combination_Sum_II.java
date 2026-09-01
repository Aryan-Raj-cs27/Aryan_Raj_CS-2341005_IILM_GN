/**
 * LeetCode 40: Combination Sum II
 * 
 * Given a collection of candidate numbers (candidates) and a target number (target), 
 * find all unique combinations in candidates where the candidate numbers sum to target.
 * 
 * Each number in candidates may only be used once in the combination. Note: The solution 
 * set must not contain duplicate combinations.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        backtrack(result, new ArrayList<>(), candidates, target, 0);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> current, int[] candidates, int remaining, int start) {
        if (remaining == 0) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        if (remaining < 0) {
            return;
        }
        
        for (int i = start; i < candidates.length; i++) {
            // Skip duplicates
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            
            current.add(candidates[i]);
            backtrack(result, current, candidates, remaining - candidates[i], i + 1);
            current.remove(current.size() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] candidates1 = {10, 1, 2, 7, 6, 1, 5};
        System.out.println(sol.combinationSum2(candidates1, 8));
        // [[1,1,6], [1,2,5], [1,7], [2,6]]
        
        int[] candidates2 = {2, 5, 2, 1, 2};
        System.out.println(sol.combinationSum2(candidates2, 5));
        // [[1,2,2], [5]]
    }
}
