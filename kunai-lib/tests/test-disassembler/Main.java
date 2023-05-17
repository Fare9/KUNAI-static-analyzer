public class Main {

    public static void test(String str, int a, double b)
    {
        System.out.println(str+a+b);
    }
    public static void main(String[] args)
    {
        int a = Integer.valueOf(args[1]);
        double b = 2.0;
        String c = "this is a test";

        int d = 10;
        int e = a+d;

        double f = b/1.0;

        test(c, e, f);
    }
}
