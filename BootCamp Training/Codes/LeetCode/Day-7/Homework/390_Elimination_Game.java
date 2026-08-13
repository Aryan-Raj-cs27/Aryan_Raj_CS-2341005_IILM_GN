class Solution {
    public int lastRemaining(int n) {
        boolean left = true;
        int remaining = n;
        int step = 1;
        int head = 1;
        
        while (remaining > 1) {
            // We update the head if we move left to right, 
            // OR if we move right to left and there are an odd number of elements left.
            if (left || remaining % 2 == 1) {
                head = head + step;
            }
            
            // Halve the remaining elements and double the step size
            remaining = remaining / 2;
            step = step * 2;
            
            // Switch direction
            left = !left;
        }
        
        return head;
    }
}