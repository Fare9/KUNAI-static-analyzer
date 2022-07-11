import java.util.*;

public class Main {
    public static int modexp(int y, int x[], int w, int n)
    {
        int R = 0, L = 0;
        int k = 0;
        int s = 1;

        while (k < w) {
            if (x[k] == 1)
                R = (s*y) % n;
            else
                R = s;
            s = R*R % n;
            L = R;
            k++;
        }

        return L;
    }

    public static void main(String[] args) throws Exception {
        int y = Integer.parseInt(args[0]);
        int x[] = {1,2,3,4,5,6,7,8,9,10};
        int w = Integer.parseInt(args[1]);
        int n = Integer.parseInt(args[2]);
        
        int ret_value = modexp(y, x, w, n);

        System.out.println("The result is: "+String.valueOf(ret_value));

        return;
    }
}