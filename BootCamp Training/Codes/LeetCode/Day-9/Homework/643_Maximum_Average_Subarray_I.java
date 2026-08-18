class Solution {
    public double findMaxAverage(int[] nums, int k) {
        long currentSum = 0;
        for (int i = 0; i < k; i++) {
            currentSum += nums[i];
        }
        
        long maxSum = currentSum;
        for (int i = k; i < nums.length; i++) {
            currentSum = currentSum + nums[i] - nums[i - k];
            maxSum = Math.max(maxSum, currentSum);
        }
        return (double) maxSum / k;
    }
}