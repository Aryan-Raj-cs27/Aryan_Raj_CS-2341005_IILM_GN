class Solution 
{
    public int maxSubarraySumCircular(int[] nums) 
    {
        int totalSum=0;
        int currMaxSum=0;
        int maxSum=nums[0];
        int currMinSum=0;
        int minSum=nums[0];
        for(int num:nums)
        {
            currMaxSum = (currMaxSum+num>num)?currMaxSum+num:num;
            maxSum = (maxSum>currMaxSum)?maxSum:currMaxSum;

            currMinSum= (currMinSum+num<num)?currMinSum+num:num;;
            minSum = (minSum<currMinSum)?minSum:currMinSum;

            totalSum+=num;
        }
        if(maxSum<0) return maxSum;
        int circularSum = totalSum - minSum;
        return (maxSum>circularSum)?maxSum:circularSum;        
    }
}
