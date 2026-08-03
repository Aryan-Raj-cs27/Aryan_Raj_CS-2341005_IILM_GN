class Main {
    static void floorAndCeil(double x) 
    {
        int floor = (int) x;

        int ceil;
        if (x == floor) 
        {
            ceil = floor;
        } else 
        {
            ceil = floor + 1;
        }

        System.out.println("Floor: " + floor);
        System.out.println("Ceil: " + ceil);
    }

    public static void main(String[] args) {
        floorAndCeil(8.3);
    }
}