package perceptron;


import java.util.Random;

import static perceptron.mathop.MathOps.*;

public class Main {

    /// Methods

    public static void shuffle(double[][] set, int[] labels){
        Random r = new Random();
        int idx1 = 0, idx2 = 0;
        for (int i = 0; i < set.length*5; i++){
            idx1 = r.nextInt(set.length);
            while(idx1 == idx2) idx2 = r.nextInt(set.length);
            double[] tmp = set[idx1].clone();
            set[idx1] = set[idx2].clone();
            set[idx2] = tmp.clone();
            int tmp2 = labels[idx1];
            labels[idx1] = labels[idx2];
            labels[idx2] = tmp2;
        }
    }

    public static boolean equal(double[] v1, double[] v2){
        for (int i = 0; i < v1.length; i++)
            if (v1[i] != v2[i]) return false;
        return true;
    }

    /**
     * Add a constant 1 at vector start
     * @param vector : vector of n elements to add const
     * @return {1, vector[0], ... , vector[n]}
     */
    public static double[] addBiasConst(double[] vector){
        double[] newVector = new double[vector.length + 1];
        newVector[0] = 1;
        for (int i = 1; i < newVector.length; i++)
            newVector[i] = vector[i - 1];
        return newVector;
    }

    /**
     * Effectue une époque et corrige les poids
     */
    public static void doEpoch(double[][] dataSet, int[] labels){

        double lR = 0.05;

        double[][] hiddenLayerDeltas = new double[numberOfInternalNeurons][D];
        double[][] outputLayerDeltas = new double[numberOfOutputNeurons][numberOfInternalNeurons + 1]; // + bias const
        // vector p, i_th element equals 1 if data is class i, else 0
        int[] probability = new int[numberOfOutputNeurons];
        double[] backPropagations = new double[numberOfOutputNeurons];

        for (int d = 0; d < dataSet.length; d++){

            double[] input = dataSet[d];
            int label = labels[d];

            // Classification

            // Hidden layer
            for (int i = 0; i < hiddenLayer.length; i++) {
                hiddenLayerPreActivations[i] = dotProduct(input, hiddenLayer[i]);
                hiddenLayerActivations[i] = sigmoid(hiddenLayerPreActivations[i]);
            }

            // Output layer
            for (int i = 0; i < outputLayer.length; i++){
                outputLayerPreActivations[i] = dotProduct(addBiasConst(hiddenLayerActivations), outputLayer[i]);
            }
            outputLayerActivations = softmax(outputLayerPreActivations);

            // probabilité selon le neurone de sortie
            for (int i = 0; i < probability.length; i++)
                probability[i] = (label == i ? 1 : 0);

            // Les backpropagations
            for (int i = 0; i < numberOfOutputNeurons; i++){
                backPropagations[i] = outputLayerActivations[i] - probability[i];
            }

            // Correction des poids

            // OL
            double[] hiddenLayersOutputs = addBiasConst(hiddenLayerActivations);
            for (int i = 0; i < numberOfOutputNeurons; i++) {
                for (int j = 0; j < hiddenLayerActivations.length + 1; j++) {
                    outputLayerDeltas[i][j] = (lR * hiddenLayersOutputs[j] * backPropagations[i]);
                    outputLayer[i][j] -= outputLayerDeltas[i][j];
                }
            }

            // HL
            double[] sum = new double[hiddenLayer.length];
            for (int i = 0; i < hiddenLayer.length; i++){
                sum[i] = 0;
                for (int j = 0; j < outputLayer.length; j++){
                    // here we take i+1_th output neuron weight because of the first one being the bias
                    sum[i] += outputLayer[j][i+1] * backPropagations[j];
                }
            }
            for (int i = 0; i < hiddenLayer.length; i++){
                for (int j = 0; j < D; j++) {
                    hiddenLayerDeltas[i][j] = lR * input[j] * sigmoidDerivated(hiddenLayerPreActivations[i]) * sum[i];
                    hiddenLayer[i][j] -= hiddenLayerDeltas[i][j];
                }
            }

        }
    }

    /**
     * Entraîne les poids pendant epMax epoques
     * @param epMax : nombre max d'époques
     */
    public static void train(int epMax){
        double[][] dataSetCopy = dataSet.clone();
        int[] labelsCopy = labels.clone();
        for (int ep = 0; ep < epMax; ep++) {
            shuffle(dataSetCopy, labelsCopy); // shuffle data and labels for each epoch
            doEpoch(dataSetCopy, labelsCopy);
        }
    }

    /**
     * Classifie l'input et renvoie la valeur de sortie
     * /!\ N'est utilisée que pour vérifier les poids après l'entraînement
     * @param input : input à classer
     * @return 1 / 0
     */
    public static int classify(double[] input){

        // Classification

        // Hidden layer
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayerPreActivations[i] = dotProduct(input, hiddenLayer[i]);
            hiddenLayerActivations[i] = sigmoid(hiddenLayerPreActivations[i]);
        }

        // Output layer
        for (int i = 0; i < outputLayer.length; i++){
            outputLayerPreActivations[i] = dotProduct(addBiasConst(hiddenLayerActivations), outputLayer[i]);
        }
        outputLayerActivations = softmax(outputLayerPreActivations);

        return (argmax(outputLayerActivations));
    }

    /// Main

    public static void main(String[] args){

        boolean randomized = true;

        double erreurs = 0;
        double essais = 1000;

        for (numberOfInternalNeurons = 3; numberOfInternalNeurons < 4; numberOfInternalNeurons++) {

            for (int e = 0; e < essais; e++) {

                // Neurons' init
                hiddenLayer = new double[numberOfInternalNeurons][D];
                hiddenLayerPreActivations = new double[numberOfInternalNeurons];
                hiddenLayerActivations = new double[numberOfInternalNeurons];

                outputLayer = new double[numberOfOutputNeurons][numberOfInternalNeurons + 1]; // + bias
                outputLayerPreActivations = new double[numberOfOutputNeurons];
                outputLayerActivations = new double[numberOfOutputNeurons];

                for (int i = 0; i < numberOfInternalNeurons; i++) {
                    for (int j = 0; j < hiddenLayer[0].length; j++)
                        hiddenLayer[i][j] = (randomized ? Math.random() : 0);
                }
                for (int i = 0; i < numberOfOutputNeurons; i++) {
                    for (int j = 0; j < outputLayer[0].length; j++)
                        outputLayer[i][j] = (randomized ? Math.random() : 0);
                }

                // training
                train(10000);

                // affichage des poids et des courbes
                //affiche();


                double[] classifications = new double[4];
                // classifie les exemples
                //System.out.println("classifications\n-------------");
                for (int i = 0; i < classifications.length; i++) {
                    double[] input = dataSet[i];
                    classifications[i] = classify(input);
                }

                if (!equal(classifications, new double[]{0, 1, 1, 0}))
                    erreurs++;
            }

            System.out.println(numberOfInternalNeurons + " neurones cachés : " + (erreurs / essais) + "% d'erreur");
            erreurs = 0;
        }
    }

    /**
     * affiche des poids et les courbes
     */
    public static void affiche(){
        // Affichage des poids
        System.out.print("Poids\n------\nw^2_1j : ");
        System.out.println("hidden layer:");
        for (int i = 0; i < hiddenLayer.length; i++){
            System.out.println("Neuron " + (i + 1) + ": ");
            for (int j = 0; j < hiddenLayer[0].length; j++){
                System.out.print(hiddenLayer[i][j]+ " , ");
            }
            System.out.println();
        }
        System.out.println("\noutput layer:");
        for (int i = 0; i < outputLayer.length; i++){
            System.out.println("Neuron " + (i + 1) + ": ");
            for (int j = 0; j < outputLayer[0].length; j++){
                System.out.print(outputLayer[i][j] + " , ");
            }
            System.out.println();
        }

        // Affichage des courbes
        System.out.println("\nCourbes de décision\n---------------");
        System.out.println("hidden layer:");
        for (int i = 0; i < hiddenLayer.length; i++){
            System.out.println("Neuron " + (i + 1) + ": ");
            for (int j = 0; j < hiddenLayer[0].length - 2; j++){
                System.out.print("(" + -hiddenLayer[i][j] / hiddenLayer[i][hiddenLayer[0].length - 1] + ") + ");
            }
            System.out.println(-hiddenLayer[i][hiddenLayer[0].length - 2] / hiddenLayer[i][hiddenLayer[0].length - 1]
                    + " x");
        }
        System.out.println("\noutput layer:");
        for (int i = 0; i < outputLayer.length; i++){
            System.out.println("Neuron " + (i + 1) + ": ");
            for (int j = 0; j < outputLayer[0].length - 2; j++){
                System.out.print("(" + -outputLayer[i][j] / outputLayer[i][outputLayer[0].length - 1] + ") + ");
            }
            System.out.println(-outputLayer[i][outputLayer[0].length - 2] / outputLayer[i][outputLayer[0].length - 1]
                    + " x");
        }

    }

    public static final int D = 3; // representation set dimension (bias const included)
    public static final double[][] dataSet = {{1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    public static final int[] labels = {0, 1, 1, 0};

    public static int numberOfInternalNeurons = 2;
    public static final int numberOfOutputNeurons = 2;

    public static double[][] hiddenLayer;
    public static double[]   hiddenLayerPreActivations;
    public static double[]   hiddenLayerActivations;
    public static double[][] outputLayer;
    public static double[]   outputLayerPreActivations;
    public static double[]   outputLayerActivations;
}
