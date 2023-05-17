import java.util.*;

public class Main {
    public static void main(String[] args) {
        int a = 0;

        System.out.println("First Basic Block");

        while (a < 3)
        {
            System.out.println("Second basic block, inside of while loop");
            a++;
        }

        System.out.println("Third Basic Block");

        if (a>=4)
        {
            System.out.println("Fourth basic block, inside of if condition");
        }

        System.out.println("Fifth Basic Block");
    }    
}
