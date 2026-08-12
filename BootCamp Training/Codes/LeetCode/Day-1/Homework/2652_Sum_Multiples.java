class Solution {
    public long findTheArrayConcVal(int[] nums) {
        long totalConcatenationValue = 0;
        int left = 0;
        int right = nums.length - 1;

        while (left <= right) {
            if (left == right) {
                // Only one element left in the middle
                totalConcatenationValue += nums[left];
            } else {
                // Two distinct elements: concatenate them mathematically
                int rightElement = nums[right];
                long multiplier = 1;
                
                // Find the appropriate power of 10 to shift the left number
                int temp = rightElement;
                while (temp > 0) {
                    multiplier *= 10;
                    temp /= 10;
                }
                
                // Add the concatenated value to the total
                totalConcatenationValue += (nums[left] * multiplier) + rightElement;
            }
            
            // Move pointers inward
            left++;
            right--;
        }

        return totalConcatenationValue;
    }
}