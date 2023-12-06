package Main;

public class Uitils {
    public static double multiply(double[] probability, int i, int j){
        if (i<=0 || j >= probability.length || i>j){
            System.out.println("输入不符合规范");
            return -1;
        }
        double result = 1;
        for (int k = i; k<=j;k++){
            result *= probability[k];
        }
        return result;

    }
}
