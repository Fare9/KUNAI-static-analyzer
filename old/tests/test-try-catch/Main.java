import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Scanner reader = new Scanner(System.in);

        int a = reader.nextInt();
        int b = 3;
        int c = 0;

        if (a == 0)
        {
            System.out.println("You gave me value 0...");
        }
        else
        {
            System.out.println("Not a bad value :D...");
        }

        try {
            c = b / a;    
            System.out.println(c);
        } catch(ArithmeticException e)
        {
            // Exception handler
            System.out.println(
                "Divided by zero operation cannot possible");
        }

        reader.close();
    }
}
