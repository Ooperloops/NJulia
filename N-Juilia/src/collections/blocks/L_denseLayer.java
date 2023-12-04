package collections.blocks;
import math.M_activation;
import math.M_array;
import java.util.Random;

public class L_denseLayer extends L_layer {
    public M_array weights;
    public M_array biases;
    private final L_layer front;
    private final L_layer back;
    private Random rd;



    public L_denseLayer(int nodes, L_layer front, L_layer back){
        rd = new Random();
        layer = new M_array(new double[nodes]);
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
    public void layerPass(){
        for(int i = 0; i < layer.array.length; i++){
            M_array iterationArr = front.layer.multiply(weights.slice(front.layer.array.length, i));
            layer.array[i] = (back == null)
                    ? M_activation.sigmoid(iterationArr.sum() + biases.array[i])
                    : M_activation.reLU(iterationArr.sum() + biases.array[i]);
        }
    }
    public M_array dLayerDx(){
        M_array arr = weights;
        M_array activations = dActivationDx();
        switch (back){
            case null:

                break;
            default:

                break;
        }
        for(int i = 0; i < activations.array.length; i++){
            for(int j = 0; j < front.layer.array.length; j++){
                arr.array[front.layer.array.length * i + j] *= activations.array[i];
            }
        }
        return arr;
    }
    private  M_array  dActivationDx(){
        M_array  arr = new  M_array(new double[layer.array.length]);
        for(int i = 0; i < layer.array.length; i++){
            M_array iterationArr = front.layer.multiply(weights.slice(front.layer.array.length, i));
            arr.array[i] = (back == null)
                    ? M_activation.dDxSigmoid(iterationArr.sum() + biases.array[i])
                    : M_activation.dDxReLU(iterationArr.sum() + biases.array[i]);
        }
        return arr;
    }
}
