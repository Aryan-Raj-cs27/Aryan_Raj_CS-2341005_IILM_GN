class Solution {
    public int subarraysDivByK(int[] nums, int k) {
        // Array to store the frequency of each remainder.
        int[] remainderCounts = new int[k];
        
        // Base case: There is 1 way to have a remainder of 0 initially (an empty prefix).
        remainderCounts[0] = 1; 

        int count = 0;
        int prefixSum = 0;

        for (int num : nums) {
            prefixSum += num;
            
            // Calculate the positive remainder
            int remainder = prefixSum % k;
            if (remainder < 0) {
                remainder += k; // Adjust negative remainders to be positive
            }

            // If we've seen this remainder before, we can form valid subarrays.
            // Add the number of times we've seen it to our total count.
            count += remainderCounts[remainder];

            // Increment the frequency of this remainder for future subarrays to use.
            remainderCounts[remainder]++;
        }

        return count;
    }
}