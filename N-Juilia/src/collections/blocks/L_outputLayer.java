package collections.blocks;

import math.M_activation;
import math.M_array;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class L_outputLayer extends L_layer{
    public Random rd;
    public L_layer front;
    private final M_activation activation;

    public L_outputLayer(int nodes, M_activation.activations activationType){
        layer = new M_array(new double[nodes]);
        unActivatedLayer = new M_array(new double[nodes]);

        this.activation = new M_activation(activationType);
    }

    public void setConnections(L_layer front){
        this.front = front;
        rd = new Random();
        biases = new M_array(new double[layer.array.length]);
        weights = new M_array(new double[front.layer.array.length * layer.array.length]);
        for(int s = 0; s < weights.array.length; s++){
            weights.array[s] = rd.nextDouble();
        }
        for(int t = 0; t < biases.array.length; t++){
            biases.array[t] = rd.nextDouble();
        }
    }

    @Override
    public void layerPass() {
        for(int i = 0; i < layer.array.length; i++){
            M_array iterationArr = front.layer.multiply(weights.slice(front.layer.array.length, i));
            unActivatedLayer.array[i] =iterationArr.sum() + biases.array[i];
            layer.array[i] = activation.activate(iterationArr.sum() + biases.array[i]);
        }
    }

    @Override
    public void backPropagateLayer(double alpha) {
        for(int i = 0; i < layer.array.length; i++){
            biases.array[i] -= layerDelta.array[i] * alpha;
            for(int j = 0; j < front.layer.array.length; j++){
                weights.array[i * front.layer.array.length + j] -=
                        layerDelta.array[i] * front.layer.array[j] * alpha;
            }
        }
    }

    public void oneNodeDeltaToError(int node, double error){
        layerDelta = new M_array(new double[layer.array.length]);
        layerDelta.array[node] = activation.activateDx(unActivatedLayer.array[node]) * error;
        System.out.println(unActivatedLayer.array[node]);
    }

}
