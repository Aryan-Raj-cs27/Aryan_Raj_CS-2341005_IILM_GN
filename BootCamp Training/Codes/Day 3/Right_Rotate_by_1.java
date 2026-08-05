public class Right_Rotate_by_1 
{
    public static void main(String[] args) 
    {
        int[] arr = {1, 2, 3, 4, 5};
        rightRotateByOne(arr);
        
        // Print the rotated array
        for (int num : arr) 
        {
            System.out.print(num + " ");
        }
    }

    public static void rightRotateByOne(int[] nums) 
    {
        if(nums==null||nums.length<=1) return;
        int n=nums.length;
        int lastElement = nums[n-1];
        for(int i=n-1;i>0;i--)
        {
            nums[i]=nums[i-1];
        }
        nums[0]=lastElement;
    }    
}
