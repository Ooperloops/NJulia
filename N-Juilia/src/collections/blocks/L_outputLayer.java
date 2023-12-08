package collections.blocks;

import math.M_activation;
import math.M_array;

import java.util.Random;

public class L_outputLayer extends L_layer{
    public Random rd;
    public L_layer front;
    public L_outputLayer(int nodes, L_layer front){
        rd = new Random();
        layer = new M_array(new double[nodes]);
        unActivatedLayer = new M_array(new double[nodes]);
        biases = new M_array(new double[nodes]);
        weights = new M_array(new double[front.layer.array.length * nodes]);
        for(int s = 0; s < weights.array.length; s++){
            weights.array[s] = rd.nextDouble();
        }
        for(int t = 0; t < biases.array.length; t++){
            biases.array[t] = rd.nextDouble();
        }
        this.front = front;
    }

    @Override
    public void layerPass() {
        for(int i = 0; i < layer.array.length; i++){
            M_array iterationArr = front.layer.multiply(weights.slice(front.layer.array.length, i));
            unActivatedLayer.array[i] =iterationArr.sum() + biases.array[i];
            layer.array[i] = M_activation.sigmoid(iterationArr.sum() + biases.array[i]);
        }
    }

    @Override
    public void backPropagateLayer() {
        for(int i = 0; i < layer.array.length; i++){
            biases.array[i] -= layerDelta.array[i];
            for(int j = 0; j < front.layer.array.length; j++){
                weights.array[i * front.layer.array.length + j] -=
                        layerDelta.array[i]
                                * front.layer.array[j];
            }
        }
    }

    public void oneNodeDeltaToError(int node, double error){
        layerDelta = new M_array(new double[layer.array.length]);
        layerDelta.array[node] = M_activation.dDxSigmoid(node) * error;
    }

}
