package src;

import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.Random;

public class IBK {
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
        knn.setKNN(30); // Set the number of neighbors (k) (root mean square of instances)
        knn.buildClassifier(newData);

        // Save the model
        File modelDir = new File("Models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        weka.core.SerializationHelper.write("Models/IBK_Model_CV.model", knn);
        System.out.println("Model saved successfully.");

        // Load the model (optional)
        IBk loadedKnn = (IBk) weka.core.SerializationHelper.read("Models/IBK_Model_CV.model");
        System.out.println("Loaded model successfully.");

        // Evaluate the model
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(newData);
        eval.crossValidateModel(loadedKnn, newData, 10, new Random(1)); // 10-fold cross-validation
        long endTime = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;

        // Print evaluation summary
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Incorrect % = " + eval.pctIncorrect());
        System.out.println("AUC = " + eval.areaUnderROC(0));
        System.out.println("Kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precision = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("F-Measure = " + eval.fMeasure(0));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
    }
}