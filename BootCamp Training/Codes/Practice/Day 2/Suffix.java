public class Suffix {
    public static void main(String[] args) 
    {        
        String word = "HelloWorld";
        System.out.println("--- SUFFIX GENERATOR ---");
        for (int i = 0; i < word.length(); i++) 
        {
            System.out.print(word.substring(i) + " ");
            
        }
    }
}