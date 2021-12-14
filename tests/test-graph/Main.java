import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Scanner reader = new Scanner(System.in);

        int a = reader.nextInt();

        int b = another_method(a);

        System.out.println(b);

        switch (a) {
            case 1:
                System.out.println("a = 1");
                break;
            case 2:
                System.out.println("a = 2");
                break;
            case 3:
                System.out.println("a = 3");
                break;
            default:
                System.out.println("a not 1,2,3");
                break;
        }

        if ((a % 2) == 0)
        {
            System.out.println("Even");
        }
        else
        {
            System.out.println("Odd");
        }

        for(int i = 0; i < 10; i++)
        {
            System.out.println("Loop number: "+String.valueOf(i));
        }

        System.out.println("Finished");
            
        return;
    }

    public static int another_method(int a)
    {
        return a+2-3/3;
    }
}