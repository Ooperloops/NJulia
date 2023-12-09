package collections.blocks;

import math.M_array;

public class L_inputLayer extends L_layer {
    public L_inputLayer(int nodes){
        layer = new M_array(new double[nodes]);
    }

    public void pass(M_array input){
        for(int i =0; i < layer.array.length; i++){
            layer.array[i] = input.array[i];
        }
    }

    @Override
    public void layerPass() {
        return;
    }

    @Override
    public void backPropagateLayer(double alpha) {
        return;
    }
}
