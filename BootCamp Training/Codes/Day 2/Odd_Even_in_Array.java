public class Odd_Even_in_Array
{
       public static void main(String[] args) 
    {
        int[] nums = {12, 35, 2, 6, 78, 97, 10, 45};
        int even = 0;
        int odd = 0;
        for (int i = 0; i < nums.length; i++) 
        {
            if (nums[i] % 2 == 0) 
            {
                even++;
            } 
            else 
            {
                odd++;
            }
        }
        System.out.println("Total Even Numbers: " + even);
        System.out.println("Total Odd Numbers: " + odd);
    }
}