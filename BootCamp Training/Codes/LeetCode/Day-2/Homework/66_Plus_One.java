class Solution {
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        
        // Traverse from right to left
        for (int i = n - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            }
            // If digit is 9, it becomes 0
            digits[i] = 0;
        }
        
        // If we reach here, it means all digits were 9 (e.g., 99 -> 100)
        int[] newNumber = new int[n + 1];
        newNumber[0] = 1;
        return newNumber;
    }
}