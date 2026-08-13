import java.util.*;
class Solution {
    public String decodeString(String s) {
        Stack<Integer> countStack = new Stack<>();
        Stack<StringBuilder> stringStack = new Stack<>();
        StringBuilder currentString = new StringBuilder();
        int k = 0;
        
        for (char ch : s.toCharArray()) {
            if (Character.isDigit(ch)) {
                // Handle multi-digit numbers (e.g., "12[a]")
                k = k * 10 + (ch - '0');
            } else if (ch == '[') {
                // Push the current k and current string to their stacks, then reset them
                countStack.push(k);
                stringStack.push(currentString);
                currentString = new StringBuilder();
                k = 0;
            } else if (ch == ']') {
                // Pop the previous string and the multiplier
                StringBuilder decodedString = stringStack.pop();
                int currentK = countStack.pop();
                
                // Append the current string k times to the decodedString
                for (int i = 0; i < currentK; i++) {
                    decodedString.append(currentString);
                }
                
                // The newly built string becomes the current string
                currentString = decodedString;
            } else {
                // Just a regular character, append it to the current string
                currentString.append(ch);
            }
        }
        
        return currentString.toString();
    }
}