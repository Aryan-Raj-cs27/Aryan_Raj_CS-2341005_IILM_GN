import java.util.HashMap;

class Solution {
    public int findMaxLength(int[] nums) {
        // Map stores (running count, first index where this count occurred)
        HashMap<Integer, Integer> map = new HashMap<>();
        
        // Base case: a count of 0 is conceptually seen just before the array starts
        map.put(0, -1);
        
        int maxLength = 0;
        int count = 0;
        
        for (int i = 0; i < nums.length; i++) {
            // Treat 1 as +1 and 0 as -1
            count += (nums[i] == 1) ? 1 : -1;
            
            if (map.containsKey(count)) {
                // We found a subarray with an equal number of 0s and 1s
                int currentLength = i - map.get(count);
                maxLength = Math.max(maxLength, currentLength);
            } else {
                // Only store the first occurrence to ensure we find the maximum length
                map.put(count, i);
            }
        }
        
        return maxLength;
    }
}