class Solution {
    public char findKthBit(int n, int k) {
        // Base case: S1 is just "0"
        if (n == 1) {
            return '0';
        }
        
        // Calculate the total length of Sn
        // The length of Sn is (2^n) - 1. We can use bit shifting for powers of 2.
        int len = (1 << n) - 1; 
        int mid = len / 2 + 1; // 1-indexed middle position
        
        // If k is exactly the middle element, it's always '1' (the added bit)
        if (k == mid) {
            return '1';
        } 
        // If k is in the left half, it's identical to the (n-1)th string
        else if (k < mid) {
            return findKthBit(n - 1, k);
        } 
        // If k is in the right half, it is the reverse and inverse of the left half
        else {
            // Find the mirrored position in the left half
            int mirroredPosition = len - k + 1;
            char mirroredBit = findKthBit(n - 1, mirroredPosition);
            
            // Return the inverted bit
            return mirroredBit == '0' ? '1' : '0';
        }
    }
}