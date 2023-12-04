package math;
import java.lang.Math;

public class M_activation {
    public static double reLU(double x){
        return (x > 0)
                ? x
                : 0;
    }
    public static double dDxReLU(double x){
        return (x > 0)
                ? 1
                : 0;
    }
    public static double sigmoid(double x){
        return (1/(1+Math.exp(-x)));
    }
    public static double dDxSigmoid(double x){
        return sigmoid(x) * (1 - sigmoid(x));
    }
}
