package math;
import javax.swing.*;
import java.lang.Math;

public class M_activation {
    public enum activations{
        RELU,
        LERELU,
        SIGMOID,
        ELU,
        TANH,
        SOFTMAX,
        ARGMAX
    }

    public final activations activationType;
    public M_activation(activations activationType){
        this.activationType = activationType;
        System.out.println(activationType);
    }
    public double activate(double z){
        return switch (activationType) {
            case RELU -> reLU(z);
            case SIGMOID -> sigmoid(z);
            case LERELU -> leReLU(z);
            case ELU -> eLu(z);
            default -> 1;
        };
    }
    public double activateDx(double z){
        return switch (activationType) {
            case RELU -> dDxReLU(z);
            case SIGMOID -> dDxSigmoid(z);
            case LERELU -> dDxLeReLU(z);
            case ELU -> dDxELu(z);
            default -> 1;
        };
    }

    private static double reLU(double x){
        return (x > 0)
                ? x
                : 0;
    }
    private static double dDxReLU(double x){
        return (x > 0)
                ? 1
                : 0;
    }
    private static double leReLU(double x){
        return (x >= 0)
                ? x
                : 0.1*x;
    }
    private static double dDxLeReLU(double x){
        return (x >= 0)
                ? 1
                : 0.1;
    }
    private static double sigmoid(double x){
        return (1/(1+Math.exp(-x)));
    }
    private static double dDxSigmoid(double x){
        return sigmoid(x) * (1 - sigmoid(x));
    }
    private static double eLu(double x){
        return (x >= 0)
                ? x
                : 0.01* (Math.exp(x) - 1);
    }
    private static double dDxELu(double x){
        return (x >= 0)
                ? 1
                : 0.01 * Math.exp(x);
    }
}
