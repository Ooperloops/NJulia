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
        M_array[] testing = {new M_array(new double[]{4, 2, 0}),
                            new M_array(new double[]{5, 3, 0}),
                            new M_array(new double[]{4, 3, 0}),
                            new M_array(new double[]{1, 1, 1}),
                            new M_array(new double[]{1, 1.5, 1}),
                            new M_array(new double[]{1, 2, 1})};

        int epochs = 50000000;
        for(int i = 0; i < epochs; i++){
            M_array input = testing[getRandomNumber(0,5)].slice(2, 0);
            M_array output = testing[getRandomNumber(0,5)].slice(1, 2);
            julia.forwardPass(input);
            julia.train(output);
        }
    }


}
