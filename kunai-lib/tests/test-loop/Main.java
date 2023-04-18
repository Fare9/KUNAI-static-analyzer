import java.*;

public class Main{
    
    void test()
    {
        int i = 0;

        while(i < 10)
        {
            System.out.println(i);
            i++;
        }
    }

    public static void main(String[] args)
    {
        Main m = new Main();

        m.test();
    }
}