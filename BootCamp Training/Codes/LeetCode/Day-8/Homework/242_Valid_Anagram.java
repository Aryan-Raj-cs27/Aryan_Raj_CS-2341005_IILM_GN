class Solution {
    public boolean isAnagram(String s, String t) {
        // If the lengths are different, they cannot be anagrams
        if (s.length() != t.length()) {
            return false;
        }
        
        // Array to store the frequency of the 26 lowercase English letters
        int[] charCounts = new int[26];
        
        // Increment for string s, decrement for string t
        for (int i = 0; i < s.length(); i++) {
            charCounts[s.charAt(i) - 'a']++;
            charCounts[t.charAt(i) - 'a']--;
        }
        
        // Check if all counts are zero
        for (int count : charCounts) {
            if (count != 0) {
                return false;
            }
        }
        
        return true;
    }
}