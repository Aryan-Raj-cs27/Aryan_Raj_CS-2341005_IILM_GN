import java.util.HashMap;
import java.util.Map;

class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefix = new HashMap<>();
        prefix.put(0, 1);
        int sum = 0;
        int count = 0;
        for (int n : nums) {
            sum += n;
            count += prefix.getOrDefault(sum - k, 0);
            prefix.put(sum, prefix.getOrDefault(sum, 0) + 1);
        }
        return count;
    }
}