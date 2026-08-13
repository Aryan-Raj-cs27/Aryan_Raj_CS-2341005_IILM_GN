class Solution {
    public String decodeAtIndex(String s, int k) {
        long size = 0;
        int n = s.length();
        
        // Step 1: Find the total length of the decoded string
        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            if (Character.isDigit(ch)) {
                size *= (ch - '0');
            } else {
                size++;
            }
        }
        
        // Step 2: Work backwards to find the k-th character
        for (int i = n - 1; i >= 0; i--) {
            char ch = s.charAt(i);
            
            // Since the string repeats, the k-th character is the same 
            // as the (k % size)-th character.
            k %= size;
            
            // If k becomes 0 and the current character is a letter, we found it!
            if (k == 0 && Character.isLetter(ch)) {
                return Character.toString(ch);
            }
            
            // If it's a digit, reduce the size by dividing
            if (Character.isDigit(ch)) {
                size /= (ch - '0');
            } 
            // If it's a letter, reduce the size by 1
            else {
                size--;
            }
        }
        
        return "";
    }
}