import java.util.HashSet;
import java.util.Set;

class Solution {
    public long maximumSubarraySum(int[] nums, int k) {
        Set<Integer> seen = new HashSet<>();
        long currentSum = 0;
        long maxSum = 0;
        int left = 0;

        for (int right = 0; right < nums.length; right++) {
            while (seen.contains(nums[right]) || (right - left + 1) > k) {
                seen.remove(nums[left]);
                currentSum -= nums[left];
                left++;
            }
            seen.add(nums[right]);
            currentSum += nums[right];
            if (right - left + 1 == k) {
                maxSum = Math.max(maxSum, currentSum);
            }
        }

        return maxSum;
    }
}