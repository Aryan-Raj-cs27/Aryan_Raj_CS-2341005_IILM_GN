import java.util.*;
public class Count_Zero 
{
    public static void main(String args[])
    {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the number of elements in the array : ");
        int n = sc.nextInt();
        int[] arr = new int [n];
        System.out.print("Enter the array elements : ");
        for(int i=0;i<n;i++)
        {
            arr[i] = sc.nextInt();
        }
        int count = 0;
        for(int i:arr )
        {
            if(i==0)
            {
                count++;
            }
        }
        System.out.println("Total number of zeros in the array : "+count);
        sc.close();
    }
}
