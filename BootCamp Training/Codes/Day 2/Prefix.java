public class Prefix 
{
    public static void main(String[] args) 
    {   
        String word = "HelloWorld";
        System.out.println("--- PREFIX GENERATOR ---");
        for (int i = 0; i < word.length(); i++) 
        {
            System.out.print(word.substring(0, i + 1) + " ");   
        }
    }
}