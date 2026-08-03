import java.util.HashMap;
import java.util.Scanner; // Required for taking user input

class Main {
    public static void main(String[] args) {
        
        // 1. Set up the Scanner and HashMap
        Scanner scanner = new Scanner(System.in);
        HashMap<Integer, String> map = new HashMap<>();

        // 2. Ask how many items to add
        System.out.print("How many items do you want to add? ");
        int numItems = scanner.nextInt();
        
        // This clears the invisible "Enter" key press left behind by nextInt()
        scanner.nextLine(); 

        System.out.println("\n--- Please enter your data ---");

        // 3. Loop to gather the input
        for (int i = 1; i <= numItems; i++) {
            System.out.print("Enter Integer key #" + i + ": ");
            int key = scanner.nextInt();
            scanner.nextLine(); // Clear the Enter key again

            System.out.print("Enter String value for key " + key + ": ");
            String value = scanner.nextLine();

            // Add the user's input to the map
            map.put(key, value);
        }

        // 4. Print the final HashMap
        System.out.println("\n--- Final HashMap Output ---");
        for (Integer key : map.keySet()) {
            System.out.println(key + " -> " + map.get(key));
        }

        // Always close the scanner to prevent memory leaks
        scanner.close();
    }
}