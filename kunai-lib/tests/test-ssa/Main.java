import java.lang.*;

public class Main {
    
    public int test_if(int a)
    {
        int x = 2;
        int y = 40;

        if (a == 0)
        {
            x = 5;
            y = 20;
        }
        else
        {
            System.out.println("This is a test!");
            y = 30;
        }

        x = x + y;

        x = x + 33;

        return x;
    }
}