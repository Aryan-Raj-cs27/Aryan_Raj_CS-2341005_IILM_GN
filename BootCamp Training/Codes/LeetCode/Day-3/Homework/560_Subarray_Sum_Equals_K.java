import java.util.HashMap;

class Solution {
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        int prefixSum = 0;
        
        // Map stores (prefix sum, frequency of that sum)
        HashMap<Integer, Integer> map = new HashMap<>();
        
        // Base case: A prefix sum of 0 has occurred 1 time (before the array starts)
        map.put(0, 1);
        
        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            
            // If we've seen a prefix sum that equals (current prefixSum - k),
            // it means the subarray between that past point and now equals k.
            int target = prefixSum - k;
            if (map.containsKey(target)) {
                count += map.get(target);
            }
            
            // Add the current prefixSum to the map, updating its frequency
            map.put(prefixSum, map.getOrDefault(prefixSum, 0) + 1);
        }
        
        return count;
    }
}