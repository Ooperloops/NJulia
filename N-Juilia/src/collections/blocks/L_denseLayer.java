package collections.blocks;
import math.M_activation;
import math.M_array;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class L_denseLayer extends L_layer {

    public M_array layerDelta;
    public L_layer front;
    public L_layer back;
    private Random rd;
    private final M_activation activation;


    public L_denseLayer(int nodes, M_activation.activations activationType){

        layer = new M_array(new double[nodes]);
        unActivatedLayer = new M_array(new double[nodes]);

        this.activation = new M_activation(activationType);
    }
    public void setConnections(L_layer front, L_layer back){
        this.front = front;
        this.back = back;
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
    public void layerPass(){
        for(int i = 0; i < layer.array.length; i++){
            M_array iterationArr = front.layer.multiply(weights.slice(front.layer.array.length, i));
            unActivatedLayer.array[i] =iterationArr.sum() + biases.array[i];
            layer.array[i] = activation.activate(iterationArr.sum() + biases.array[i]);
        }
    }

    //-----------------------------------------------------------------------------------------------------
    // Backpropgation methods
    //-----------------------------------------------------------------------------------------------------
    @Override
    public void backPropagateLayer(double alpha) {
        for(int i = 0; i < layer.array.length; i++){
            biases.array[i] -= layerDelta.array[i]*alpha;
            for(int j = 0; j < front.layer.array.length; j++){
                weights.array[i * front.layer.array.length + j] -=
                        layerDelta.array[i] * front.layer.array[j]*alpha;
            }
        }
    }

    public void nodeDeltaToError(){
        layerDelta = new M_array(new double[layer.array.length]); // what is used for adjusting layer weights
        for(int i = 0; i < layerDelta.array.length; i++){
            for(int j = 0; j < back.layerDelta.array.length; j++){
                double delta = back.layerDelta.array[j]
                        * back.weights.array[j * back.layerDelta.array.length + i]
                        * activation.activateDx(unActivatedLayer.array[i]);
                layerDelta.array[i] += delta;

            }
        }
    }
}
