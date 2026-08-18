class Solution {
    public int characterReplacement(String s, int k) {
        int[] charCounts = new int[26];
        int left = 0;
        int maxFrequency = 0;
        int maxLength = 0;

        for (int right = 0; right < s.length(); right++) {
            int currentCharIdx = s.charAt(right) - 'A';
            charCounts[currentCharIdx]++;
            maxFrequency = Math.max(maxFrequency, charCounts[currentCharIdx]);
            while ((right - left + 1) - maxFrequency > k) {
                int leftCharIdx = s.charAt(left) - 'A';
                charCounts[leftCharIdx]--;
                left++;
            }
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }
}