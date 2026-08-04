public class Greater_than_10 
{
    public static void main(String[] args) 
    {
        int[] nums = {5, 12, 7, 18, 3, 25, 10};
        System.out.println("Numbers greater than 10:");
        for (int i=0; i<nums.length; i++) 
        {
            if (nums[i]>10) 
            {
                System.out.print(nums[i]+" ");
            }
        }
    }
    
}
