class Reverse_Traversal
{
    public static void main(String[] args)
    {
        int[] nums = {10,20,30,40,50};
        System.out.println("Array in reverse order :");
        for(int i = nums.length-1;i>=0;i--)
        {
            System.out.print(nums[i]+" ");
        }
    }
}