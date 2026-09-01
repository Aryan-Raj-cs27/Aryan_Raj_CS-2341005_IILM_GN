/**
 * LeetCode 39: Combination Sum
 * 
 * Given an array of distinct integers candidates and a target integer target, 
 * return a list of all unique combinations of candidates where the chosen numbers sum to target.
 * 
 * You may return the combinations in any order. The same number may be chosen from candidates 
 * an unlimited number of times. Two combinations are unique if the frequency of at least one 
 * of the chosen numbers is different.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
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
            current.add(candidates[i]);
            backtrack(result, current, candidates, remaining - candidates[i], i);
            current.remove(current.size() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        int[] candidates1 = {2, 3, 6, 7};
        System.out.println(sol.combinationSum(candidates1, 7));
        // [[2,2,3], [7]]
        
        int[] candidates2 = {2, 3, 5};
        System.out.println(sol.combinationSum(candidates2, 8));
        // [[2,2,2,2], [2,3,3], [3,5]]
        
        int[] candidates3 = {2};
        System.out.println(sol.combinationSum(candidates3, 1));
        // []
    }
}
