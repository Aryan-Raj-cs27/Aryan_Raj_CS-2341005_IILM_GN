class Solution {
    public int numOfSubarrays(int[] arr, int k, int threshold) {
        int targetSum = k * threshold;
        int currentSum = 0;
        int validSubarrays = 0;
        for (int i = 0; i < k; i++) {
            currentSum += arr[i];
        }
        if (currentSum >= targetSum) {
            validSubarrays++;
        }
        for (int i = k; i < arr.length; i++) {
            currentSum = currentSum + arr[i] - arr[i - k];
            if (currentSum >= targetSum) {
                validSubarrays++;
            }
        }
        
        return validSubarrays;
    }
}