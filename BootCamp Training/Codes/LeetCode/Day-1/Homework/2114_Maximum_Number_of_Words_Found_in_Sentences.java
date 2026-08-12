class Solution {
    public int mostWordsFound(String[] sentences) {
        int maxWords = 0;
        
        for (String sentence : sentences) {
            int currentWordCount = 1; // A sentence has at least one word
            
            // Count the spaces in the current sentence
            for (int i = 0; i < sentence.length(); i++) {
                if (sentence.charAt(i) == ' ') {
                    currentWordCount++;
                }
            }
            
            // Update maxWords if the current sentence has more words
            maxWords = Math.max(maxWords, currentWordCount);
        }
        
        return maxWords;
    }
}