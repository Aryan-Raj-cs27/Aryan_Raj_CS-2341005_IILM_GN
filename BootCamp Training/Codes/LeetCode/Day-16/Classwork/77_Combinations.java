/**
 * LeetCode 77: Combinations
 * 
 * Given two integers n and k, return all possible combinations of k numbers 
 * chosen from the range [1, n].
 * 
 * You may return the combinations in any order.
 */

import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), n, k, 1);
        return result;
    }
    
    private void backtrack(List<List<Integer>> result, List<Integer> current, int n, int k, int start) {
        if (current.size() == k) {
            result.add(new ArrayList<>(current));
            return;
        }
        
        for (int i = start; i <= n; i++) {
            current.add(i);
            backtrack(result, current, n, k, i + 1);
            current.remove(current.size() - 1);
        }
    }
    
    public static void main(String[] args) {
        Solution sol = new Solution();
        
        System.out.println(sol.combine(4, 2));
        // [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
        
        System.out.println(sol.combine(1, 1));
        // [[1]]
    }
}
