class Solution {
    public int maxSubarraySum(int[] arr, int k) 
    {
        int n = arr.length;
        if (n < k) 
        {
            return -1; 
        }
        int currentSum = 0;
        int maxSum = 0;
        for (int i = 0; i < k; i++) 
        {
            currentSum += arr[i];
        }
        maxSum = currentSum;
        
        for (int i = k; i < n; i++) 
        {
            currentSum = currentSum + arr[i] - arr[i - k];
            maxSum = Math.max(maxSum, currentSum);
        }
        return maxSum;
    }
}