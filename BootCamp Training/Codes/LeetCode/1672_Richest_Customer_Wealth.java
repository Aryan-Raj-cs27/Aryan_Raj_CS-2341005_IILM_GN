class Solution {
    public int maximumWealth(int[][] accounts) {
    int maxWealth = 0;
    for (int[] customer : accounts) 
    {
        int currentWealth = 0;
        for (int bank : customer)
        {
            currentWealth += bank;
        } 
        maxWealth = Math.max(maxWealth, currentWealth);
    }
    return maxWealth;
}
}