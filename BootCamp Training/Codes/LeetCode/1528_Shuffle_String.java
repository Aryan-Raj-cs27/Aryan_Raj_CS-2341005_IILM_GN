class Solution 
{
    public String restoreString(String s, int[] indices) 
    {
        int n = s.length();
        char[] a=new char[n];
        for(int i=0;i<n;i++)
        {
            a[indices[i]]=s.charAt(i);
        }
        return new String(a);
    }
}
