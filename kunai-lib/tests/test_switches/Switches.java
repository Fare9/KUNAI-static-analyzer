import java.lang.*;

public class Switches
{
    public void switch1(int a)
    {
        switch(a)
        {
        case 5:
            System.out.println("Value is 5");
            break;
        case 6:
            System.out.println("Value is 6");
            break;
        case 7:
            System.out.println("Value is 7");
            break;
        default:
            System.out.println("No value match...");
        }
    }

    public void switch2(int a)
    {
        switch(a)
        {
        case 'a':
            System.out.println("Value is value character 'a'");
            break;
        case 'z':
            System.out.println("Value is value character 'z'");
            break;
        case 'q':
            System.out.println("Value is value character 'q'");
            break;
        default:
            System.out.println("No value match...");
        }
    }

    public void switch3(int a)
    {
        switch(a)
        {
        case 5:
            System.out.println("Value is 5");
            break;
        case 6:
            System.out.println("Value is 6");
            break;
        case 7:
            System.out.println("Value is 7");
            break;
        }
    }

    public void switch4(char a)
    {
        switch(a)
        {
        case 'a':
            System.out.println("Value is value character 'a'");
            break;
        case 'z':
            System.out.println("Value is value character 'z'");
            break;
        case 'q':
            System.out.println("Value is value character 'q'");
            break;
        }
    }

    public int switch5(int a)
    {
        int c = 5;
        int d = 0;
        switch(a)
        {
        case 52:
            d = 32 + c;
            break;
        case 86:
            d = 11 + c;
            break;
        case 109:
            d = 2 + 2;
            break;
        case 1:
            System.out.println("Value is equals to 1");
            break;
        }

        return d;
    }

    public static void main(String[] args)
    {
        Switches s = new Switches();

        int number = Integer.parseInt(args[0]);
        char c = args[1].charAt(0);

        s.switch1(number);
        s.switch2(c);
        s.switch3(number);
        s.switch4(c);
        s.switch5(number);
    }
}