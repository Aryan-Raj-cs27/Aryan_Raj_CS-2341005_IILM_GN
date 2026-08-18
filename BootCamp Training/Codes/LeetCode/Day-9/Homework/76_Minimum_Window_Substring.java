class Solution {
    public String minWindow(String s, String t) {
        if (s == null || t == null || s.length() == 0 || t.length() == 0 || s.length() < t.length()) {
            return "";
        }
        int[] charCount = new int[128];
        for (char c : t.toCharArray()) {
            charCount[c]++;
        }

        int left = 0, right = 0;
        int minLeft = 0, minLen = Integer.MAX_VALUE;
        int required = t.length();

        while (right < s.length()) {
            if (charCount[s.charAt(right)] > 0) {
                required--;
            }
            charCount[s.charAt(right)]--;
            right++;
            while (required == 0) {
                if (right - left < minLen) {
                    minLen = right - left;
                    minLeft = left;
                }
                charCount[s.charAt(left)]++;
                if (charCount[s.charAt(left)] > 0) {
                    required++;
                }
                
                left++;
            }
        }

        return minLen == Integer.MAX_VALUE ? "" : s.substring(minLeft, minLeft + minLen);
    }
}