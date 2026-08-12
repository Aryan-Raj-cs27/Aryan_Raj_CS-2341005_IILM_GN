class Solution 
{
    public int majorityElement(int[] nums) 
    {
        int majority_count=0;
        int element =0;
        for(int x : nums)
        {
            if(majority_count==0)
            {
                element=x;
               majority_count=1;
            }
            else
            {
                if(element==x)
                {
                    majority_count++;
                }
                else
                {
                    majority_count--;
                }
            }
        }
        return element;
        } 
        
}
