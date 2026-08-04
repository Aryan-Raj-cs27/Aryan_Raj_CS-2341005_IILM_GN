class Solution {
    public int thirdMax(int[] nums) {
        Long fmax = null;
        Long smax = null;
        Long tmax = null;
        for (int i : nums) 
        {
            if ((fmax != null && i == fmax) || 
                (smax != null && i == smax) || 
                (tmax != null && i == tmax)) {
                continue;
            }
            if (fmax == null || i > fmax) 
            {
                tmax = smax;
                smax = fmax;
                fmax = (long) i;
            } else if (smax == null || i > smax) {
                tmax = smax;
                smax = (long) i;
            } else if (tmax == null || i > tmax) {
                tmax = (long) i;
            }
        }
        return tmax==null?fmax.intValue():tmax.intValue();
    }
}
