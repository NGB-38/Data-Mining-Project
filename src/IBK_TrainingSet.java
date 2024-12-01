package src;

import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class IBK_TrainingSet {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Dataset/Titanic.arff");
        Instances dataset = source.getDataSet();

        // Convert Survived attribute to nominal
        NumericToNominal numToNom = new NumericToNominal();
        numToNom.setAttributeIndices("1"); // First attribute (Survived)
        numToNom.setInputFormat(dataset);
        Instances newData = Filter.useFilter(dataset, numToNom);

        // Set the class attribute (target variable)
        newData.setClassIndex(0); // 'Survived' is the first attribute

        // Build IBK classifier
        IBk knn = new IBk();
        knn.setKNN(30); // Set the number of neighbors (k)
        knn.buildClassifier(newData);

        // Save the model
        File modelDir = new File("Models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        weka.core.SerializationHelper.write("Models/IBK_Model_TS.model", knn);
        System.out.println("Model saved successfully.");

        // Load the model (optional)
        IBk loadedKnn = (IBk) weka.core.SerializationHelper.read("Models/IBK_Model_TS.model");
        System.out.println("Loaded model successfully.");

        // Evaluate the model using training data (not cross-validation)
        Evaluation eval = new Evaluation(newData);
        eval.evaluateModel(knn, newData); // Evaluate the model on the same dataset
        System.out.println("\nTraining dataset evaluation results:");
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
    }
}
