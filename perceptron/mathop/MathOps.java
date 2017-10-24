package perceptron.mathop;

import java.util.Arrays;

import static java.lang.Math.exp;

// Operations mathématiques
public class MathOps {

    public static double sigmoid(double x){
        return (1f/(1f+Math.exp(-x)));
    }

    public static double reversedSigmoid(double x) {
        return (1f/(1f+Math.exp(x)));
    }

    public static double sigmoidDerivated(double x){
        double result = sigmoid(x)*(1 - sigmoid(x));
        return result;
    }

    public static double[] softmax(double[] values){
        double[] softmax = new double[values.length];
        double[] exps = new double[values.length];

        for (int i = 0; i < values.length; i++)
            exps[i] = exp(values[i]);
        double sumExps = sum(exps);

        for (int i = 0; i < values.length; i++) {
            /* if values are too negative, exponentials equal 0 and the calculation
             for the softmax by the basic formulae is impossible, so in case the exp equal 0,
             the softmax is caculated via another formulae */

            double tmp = 0;
            for (int j = 0; j < values.length; j++)
                tmp += exp(values[j] - values[i]);
            softmax[i] = 1d/tmp;
            /*
            if (exps[i] == 0){
                double tmp = 0;
                for (int j = 0; j < values.length; j++)
                    tmp += exp(values[j] - values[i]);
                softmax[i] = 1d/tmp;
            }
            // if values are large enough
            else {
                softmax[i] = exps[i] / sumExps;
            }*/
        }
        return softmax;
    }

    public static int argmax(double[] vector){
        if (vector.length > 0) {
            int index = 0;

            for (int i = 0; i < vector.length; i++) {
                if (vector[i] > vector[index])
                    index = i;
            }

            return index;
        } else {
            return -1;
        }
    }

    public static int treshold(double value, double t){
        if (value >= t) return 1;
        else            return 0;
    }

    public static int threshold(double value){
        return treshold(value, 0);
    }

    public static double dotProduct(double[] v1, double[] v2){
        if (v1.length == v2.length) {
            double product = 0;
            for (int i = 0; i < v1.length; i++)
                product += v1[i] * v2[i];
            return product;
        } else {
            throw new RuntimeException();
        }
    }

    public static double sum(double[] vector){
        double sum = 0;
        for (double d : vector) sum += d;
        return sum;
    }
}
