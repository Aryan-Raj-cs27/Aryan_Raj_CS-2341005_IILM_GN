import java.util.ArrayList;
import java.util.List;
class Solution 
{
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) 
    {
        int maxcandy=0;
        for(int candy:candies)
        {
            if(candy>maxcandy) maxcandy=candy;
        }
        List<Boolean> result= new ArrayList<>();
        for(int candy:candies)
        {
            result.add((candy+extraCandies)>=maxcandy ? true : false);
        }
        return result;        
    }
}