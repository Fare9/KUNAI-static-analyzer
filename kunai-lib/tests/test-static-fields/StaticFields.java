import java.lang.*;

public class StaticFields
{
    public static final int field_int = 5;
    public static final String field_str = "this is a test";
    public static final byte[] field_byte_array = {1,2,3,4,5,6};

    public static void main(String[] args)
    {
        System.out.println(StaticFields.field_int);
        System.out.println(StaticFields.field_str);

        for(int i = 0; i < 6; i++)
        {
            System.out.println(StaticFields.field_byte_array[i]);
        }
    }
}