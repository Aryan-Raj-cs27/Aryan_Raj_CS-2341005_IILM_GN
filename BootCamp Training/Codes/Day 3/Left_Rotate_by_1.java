public class Left_Rotate_by_1 
{
    public static void main(String[] args) 
    {
        int[] arr = {1, 2, 3, 4, 5};
        leftRotateByOne(arr);
        
        // Print the rotated array
        for (int num : arr) 
        {
            System.out.print(num + " ");
        }
    }

    public static void leftRotateByOne(int[] nums) 
    {
        if(nums==null||nums.length<=1) return;
        int n=nums.length;
        int firstElement = nums[0];
        for(int i=0;i<n-1;i++)
        {
            nums[i]=nums[i+1];
        }
        nums[n-1]=firstElement;
    }    
}