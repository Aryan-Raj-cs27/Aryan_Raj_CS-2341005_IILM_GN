class Solution {
    public int numberOfSubarrays(int[] nums, int k) 
    {
        int[] prefixCounts = new int[nums.length + 1];
        prefixCounts[0] = 1; 
        
        int currentOddCount = 0;
        int totalNiceSubarrays = 0;
        
        for (int num : nums) 
        {
            currentOddCount += (num % 2);
            if (currentOddCount >= k) 
            {
                totalNiceSubarrays += prefixCounts[currentOddCount - k];
            }
            prefixCounts[currentOddCount]++;
        }
        return totalNiceSubarrays;
    }
}