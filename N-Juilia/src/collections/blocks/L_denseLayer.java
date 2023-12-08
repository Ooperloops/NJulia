package collections.blocks;
import math.M_activation;
import math.M_array;
import java.util.Random;

public class L_denseLayer extends L_layer {

    public M_array layerDelta;
    private final L_layer front;
    private final L_layer back;
    private Random rd;



    public L_denseLayer(int nodes, L_layer front, L_outputLayer back){
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
        this.back = back;
    }
    @Override
    public void layerPass(){
        for(int i = 0; i < layer.array.length; i++){
            M_array iterationArr = front.layer.multiply(weights.slice(front.layer.array.length, i));
            unActivatedLayer.array[i] =iterationArr.sum() + biases.array[i];
            layer.array[i] = (back == null)
                    ? M_activation.sigmoid(iterationArr.sum() + biases.array[i])
                    : M_activation.reLU(iterationArr.sum() + biases.array[i]);
        }
    }

    //-----------------------------------------------------------------------------------------------------
    // Backpropgation methods
    //-----------------------------------------------------------------------------------------------------
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

    public void nodeDeltaToError(){
        layerDelta = new M_array(new double[layer.array.length]); // what is used for adjusting layer weights
        for(int i = 0; i < layerDelta.array.length; i++){
            for(int j = 0; j < back.layerDelta.array.length; j++){
                double delta = back.layerDelta.array[j]
                        * back.weights.array[j * back.layerDelta.array.length + i]
                        * M_activation.dDxReLU(unActivatedLayer.array[i]);
                layerDelta.array[i] += delta;
            }
        }
    }
}
