package tests;
import collections.N_perceptron;
import math.M_array;

import java.util.Arrays;
import java.util.Random;

public class testing {
    public static int getRandomNumber(int min, int max) {
        Random random = new Random();
        return random.nextInt(max - min) + min;
    }
    public static void main(String args[]) {
        N_perceptron julia = new N_perceptron();
        M_array[] testing = {new M_array(new double[]{1,0}),
                new M_array(new double[]{2,0}),
                new M_array(new double[]{3,0}),
                new M_array(new double[]{7,0}),
                new M_array(new double[]{8,0}),
                new M_array(new double[]{5,0}),
                new M_array(new double[]{10,1}),
                new M_array(new double[]{11,1}),
                new M_array(new double[]{12,1}),
                new M_array(new double[]{19,1}),
                new M_array(new double[]{20,1}),
                new M_array(new double[]{14,1}),};

        int epochs = 500000;
        for(int i = 0; i < epochs; i++){
            int a = getRandomNumber(0,testing.length -1);
            M_array input = testing[a].slice(1, 0);
            M_array output = testing[a].slice(1, 1);
            julia.forwardPass(input);

            julia.train(output);
        }
    }


}
