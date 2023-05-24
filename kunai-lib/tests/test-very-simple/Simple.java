public class Simple {

    private short test_field = 2;

    public int my_add(int a, int b)
    {
        int array[2];

        int c = a + b;
        
        int d = c * a;

        int e = d / b;

        short f = test_field;
        
        test_field = f;

        array[0] = 1;

        int x = a;

        if (c == e)
            x = c;
        else
            x = e;

        e = x;

        return e;
    }

    public static void main(String[] args) throws Exception {
        Simple s = new Simple();

        int a = Integer.parseInt(args[0]);
        int b = Integer.parseInt(args[1]);
        int c = s.my_add(a, b);

        System.out.println(c);
    }
}