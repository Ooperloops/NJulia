package collections.blocks;
import math.M_array;

public abstract class L_layer {
    public M_array layer;
    public M_array weights;
    public M_array biases;
    public M_array layerDelta;
    protected M_array unActivatedLayer;
    public abstract void layerPass();
    public abstract void backPropagateLayer();
}
