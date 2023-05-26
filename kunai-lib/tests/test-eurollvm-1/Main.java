import java.lang.*;

public class Main
{
    private final double pi = 3.14;

    private double d;

    private int r;

    public double test(int a)
    {
        String example = "The result is equals to: ";

        int c = a * 40;

        c = c + 4;

        r = c;

        d = 2 * pi * r;

        System.out.println(example + Double.toString(d));

        return d;
    }
}