class Solution {
    public int hIndex(int[] citations) {
        int n = citations.length;
        int[] counts = new int[n + 1];
        
        for (int c : citations) {
            if (c >= n) {
                counts[n]++;
            } else {
                counts[c]++;
            }
        }
        
        int totalPapers = 0;
        for (int i = n; i >= 0; i--) {
            totalPapers += counts[i];
            if (totalPapers >= i) {
                return i;
            }
        }
        
        return 0;
    }
}