package math;

public class M_array {
    public double[] array;
    public M_array(double[] arr){
        array = arr;
    }
    public void multiplyBy(M_array arr){
        for(int i = 0; i < array.length; i++){
            for (double v : arr.array) {
                array[i] *= v;
            }
        }
    }
    public void addBy(M_array arr){
        if(array.length != arr.array.length) System.out.println("!!: shape error");
        for(int i = 0; i < array.length; i++){
            array[i] += arr.array[i];
        }
    }
    public M_array multiply(M_array arr){
        M_array newArray = new M_array(new double[arr.array.length]);
        for(int i = 0; i < array.length; i++){
            for (double v : arr.array) {
                newArray.array[i] = array[i] * v;
            }
        }
        return newArray;

    }
    public M_array add(M_array arr){
        if(array.length != arr.array.length) System.out.println("!!: shape error");
        M_array newArray = new M_array(new double[arr.array.length]);
        for(int i = 0; i < array.length; i++){
            newArray.array[i] = array[i] += arr.array[i];
        }

        return newArray;
    }
    public M_array dotProduct(M_array arr){
        M_array newArray = new M_array(new double[arr.array.length]);
        for(int i = 0; i < array.length; i++){
            newArray.array[i] = array[i] * arr.array[i];
        }
        return newArray;
    }
    public double sum(){
        double a = 0;
        for(double s : array){
            a += s;
        }
        return a;
    }

    public M_array slice(int slice, int iteration){
        M_array arr;
        double[] arr1 = new double[slice];
        int section =
                (iteration * slice);
        if(section >= array.length) return null;

        for(int s = 0; s < slice; s++){
            arr1[s] = array[section + s];
        }
        arr = new M_array(arr1);
        return arr;
    }
}

