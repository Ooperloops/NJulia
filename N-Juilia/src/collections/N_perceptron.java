package collections;

import collections.blocks.L_denseLayer;
import collections.blocks.L_inputLayer;
import collections.blocks.L_outputLayer;
import math.M_array;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class N_perceptron {
    public L_inputLayer inputLayer;
    public L_outputLayer finalLayer;

    public N_perceptron(){
        inputLayer = new L_inputLayer(2);
        finalLayer = new L_outputLayer(1, inputLayer);
    }


    public void forwardPass(M_array input){
        inputLayer.pass(input);
        finalLayer.layerPass();
    }
    public void train(M_array expectedOutput){
        M_array error;
        for(int i = 0; i < expectedOutput.array.length; i++){
            error = finalLayer.layer.subtract(expectedOutput);
            System.out.println(Double.toString(error.array[i]));
            double err = error.sum() * 2;

            finalLayer.oneNodeDeltaToError(i, error.array[i] * 0.002);
            finalLayer.backPropagateLayer();
        }
    }

}
