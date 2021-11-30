import java.util.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Scanner reader = new Scanner(System.in);

        int a = reader.nextInt();

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
}