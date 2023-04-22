import java.lang.*;

public class Main {
    private final double pi = 3.14;

    private double e_double_field;
    private int a_int_field;
    private int a_int_neg_field;

    private long example_long = 4;

    private double example_double = 4.0;

    private double assignment_double;
    private float assignment_float;
    private long assignment_long;
    private int assignment_int;
    private short assignment_short;
    private char assignment_char;
    private byte assignment_byte;

    public void test(float a, int b, short c)
    {
	assignment_double = 69.0;
	assignment_float = 69.0f;
	assignment_long = 69;
	assignment_int = 69;
	assignment_short = 69;
	assignment_char = 'E';
	assignment_byte = 69;

        int pi_int = (int)pi;

        float e = a * b;
        
        double e_double = (double)e;

        e_double_field = e_double;

        int a_int = (int)a;

        a_int_field = a_int;

        int a_int_neg = -a_int;

        a_int_neg_field = a_int_neg;

	double test_double = example_double;

	test_double = test_double + 1;

	example_double = test_double;

	long test_long = example_long;

	test_long = test_long + 1;

	example_long = test_long;
    }
}
