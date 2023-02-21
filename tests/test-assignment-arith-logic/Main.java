import java.util.*;

public class Main {
    static int field_int;
    static boolean field_boolean = true;
    static int[] field_int_array = {1,2,3,4,5,6,7,8,9};
    public static void main(String[] args) throws Exception {
        // Your code here!
        Scanner reader = new Scanner(System.in);

        int a = reader.nextInt();
        int b = 3;
        int c = a + b + 7;

        float d = 3.14f;
        float e = d * 2;

        String str = "ojete de vaca";
        
        field_int = reader.nextInt();

        boolean my_field_boolean = field_boolean;
        int my_field_int = field_int;
        
        long h = (long)c;

        boolean test_cmpf = d > e;

        System.out.println(a);
        System.out.println(b);
        System.out.println(c);
        System.out.println(d);
        System.out.println(e);
        System.out.println(str);
        System.out.println("Test case");
        System.out.println(my_field_int);
        System.out.println(my_field_boolean);
        System.out.println(h);
        System.out.println(test_cmpf);

        if (c > 20)
        {
            System.out.println("c > 20");
        }else
        {
            System.out.println("c <= 20");
        }

        if (c <= 0)
        {
            System.out.println("c <= 0");
        }
        else
        {
            System.out.println("c > 0");
        }

        reader.close();
    }
}
