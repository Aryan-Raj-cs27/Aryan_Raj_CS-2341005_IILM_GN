import java.util.HashMap;

class Solution {
    public boolean checkSubarraySum(int[] nums, int k) {
        // Map stores (prefixSum % k, first index where this remainder occurred)
        HashMap<Integer, Integer> modSeen = new HashMap<>();
        
        // Base case: to handle a valid subarray that starts from index 0
        modSeen.put(0, -1);
        
        int prefixSum = 0;
        
        for (int i = 0; i < nums.length; i++) {
            prefixSum += nums[i];
            
            // Calculate the remainder of the current prefix sum
            int mod = prefixSum % k;
            
            // If we've seen this remainder before
            if (modSeen.containsKey(mod)) {
                // Check if the subarray length is at least 2
                if (i - modSeen.get(mod) >= 2) {
                    return true;
                }
            } else {
                // Only store the first occurrence to ensure we check the longest possible subarrays
                modSeen.put(mod, i);
            }
        }
        
        return false;
    }
}