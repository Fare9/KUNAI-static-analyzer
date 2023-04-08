public class Simple {
    public int my_add(int a, int b)
    {
        int c = a + b;
        return c;
    }

    public static void main(String[] args) throws Exception {
        Simple s = new Simple();

        int a = Integer.parseInt(args[0]);
        int b = Integer.parseInt(args[1]);
        int c = s.my_add(a, b);

        System.out.println(c);
    }
}