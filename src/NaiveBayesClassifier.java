package src;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.SerializationHelper;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.ObjectOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class NaiveBayesClassifier {
    public static void train(String inputArffPath, String outputModelPath) throws Exception {
        // Load ARFF
        DataSource source = new DataSource(inputArffPath);
        Instances data = source.getDataSet();

        // Set class index (Survived attribute)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Apply NumericToNominal filter if necessary
        if (data.classAttribute().isNumeric()) {
            System.out.println("Converting numeric class to nominal...");
            NumericToNominal convert = new NumericToNominal();
            convert.setAttributeIndices("last"); // Specify the class attribute
            convert.setInputFormat(data);
            data = Filter.useFilter(data, convert);
        }

        // Build NaiveBayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);

        // Save the model
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(outputModelPath));
        oos.writeObject(nb);
        oos.close();
        System.out.println("Model saved to: " + outputModelPath);
    }


    public static void main(String[] args) {
        try {
            train("Dataset/Titanic.arff", "naiveBayes2.model");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
