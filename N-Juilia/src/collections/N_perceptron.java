package collections;

import collections.blocks.L_denseLayer;
import collections.blocks.L_inputLayer;
import collections.blocks.L_outputLayer;
import math.M_activation;
import math.M_array;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class N_perceptron {
    public L_inputLayer inputLayer;
    public L_denseLayer d1;
    public L_outputLayer finalLayer;
    public double alpha = 0.02;

    public N_perceptron(){
        inputLayer = new L_inputLayer(1);
        d1 = new L_denseLayer(1, M_activation.activations.ELU);
        finalLayer = new L_outputLayer(1, M_activation.activations.SIGMOID);
        d1.setConnections(inputLayer,finalLayer);
        finalLayer.setConnections(d1);
    }


    public void forwardPass(M_array input){
        inputLayer.pass(input);
        d1.layerPass();
        finalLayer.layerPass();
    }
    public void train(M_array expectedOutput){
        M_array error;
        for(int i = 0; i < expectedOutput.array.length; i++){
            error = finalLayer.layer.subtract(expectedOutput);

            double err = error.array[i];

            finalLayer.oneNodeDeltaToError(i, err);
            finalLayer.backPropagateLayer(alpha);
            d1.nodeDeltaToError();
            d1.backPropagateLayer(alpha);
            System.out.println(Arrays.toString(finalLayer.layerDelta.array));
            System.out.println(Double.toString(finalLayer.weights.array[0]) + "=W");
            System.out.println(Double.toString(finalLayer.biases.array[0]) + "=B");
            System.out.println(Double.toString(err) + "=E");
        }
    }

}
