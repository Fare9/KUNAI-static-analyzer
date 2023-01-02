import java.util.*;

public class Main {
    public static void main(String[] args) {
        // new objects of objects
        String str_object = new String("test1");
        
        // new arrays
        int[] int_array = {1,2,3};
        String[] str_array = {"1","2","3"};

        int[][] int_multidimensial_array = new int[3][3];

        int_multidimensial_array[0][0] = 0;

        System.out.println(str_object);

        for(int i = 0; i < int_array.length; i++)
            System.out.println(int_array[i]);
        
        for(int i = 0; i < str_array.length; i++)
            System.out.println(str_array[i]);

        System.out.println(int_multidimensial_array[0][0]);
    }
}