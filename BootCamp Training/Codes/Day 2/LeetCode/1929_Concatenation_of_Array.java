class a_1929_Concatenation_of_Array {
    public static void main(String[] args) {
        int[] nums = {1, 2, 3};
        int[] result = getConcatenation(nums);
        
        System.out.print("Concatenated Array: ");
        for (int num : result) {
            System.out.print(num + " ");
        }
    }
    
    public static int[] getConcatenation(int[] nums) {
        int n = nums.length;
        int[] result = new int[2 * n];
        
        for (int i = 0; i < n; i++) {
            result[i] = nums[i];
            result[i + n] = nums[i];
        }
        
        return result;
    }
}