package src;

import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.File;

public class IBK {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Dataset/Titanic.arff");
        Instances dataset = source.getDataSet();

        // Convert Survived to nominal
        weka.filters.unsupervised.attribute.NumericToNominal convert = new weka.filters.unsupervised.attribute.NumericToNominal();
        convert.setAttributeIndices("1"); // First attribute (Survived)
        convert.setInputFormat(dataset);
        dataset = weka.filters.Filter.useFilter(dataset, convert);

        // Set the class attribute (target variable)
        dataset.setClassIndex(0); // 'Survived' is the first attribute

        // Build IBK classifier
        weka.classifiers.lazy.IBk knn = new weka.classifiers.lazy.IBk();
        knn.setKNN(2); // Set the number of neighbors (k)
        knn.buildClassifier(dataset);

        // Evaluate the model
        weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(dataset);
        eval.crossValidateModel(knn, dataset, 10, new java.util.Random(1)); // 10-fold cross-validation

        // Print evaluation summary
        System.out.println(eval.toSummaryString("=== Evaluation Summary ===", true));
        System.out.println("Precision: " + eval.weightedPrecision());
        System.out.println("Recall: " + eval.weightedRecall());
        System.out.println("F-Measure: " + eval.weightedFMeasure());

        // Save the model
        weka.core.SerializationHelper.write("Models/IBK_Model.model", knn);
        System.out.println("Model saved successfully.");

        // Load the model (optional)
        weka.classifiers.lazy.IBk loadedKnn = (weka.classifiers.lazy.IBk) weka.core.SerializationHelper.read("Models/IBK_Model.model");
        System.out.println("Loaded model successfully.");
    }
}
