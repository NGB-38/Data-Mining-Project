package src;

import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class IBK_TrainingSet {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Dataset/TitanicNom.arff");
        Instances dataset = source.getDataSet();

        // Set the class attribute (target variable)
        dataset.setClassIndex(0); // 'Survived' is the first attribute

        // Build IBK classifier
        IBk knn = new IBk();
        knn.setKNN(30); // Set the number of neighbors (k)
        knn.buildClassifier(dataset);

        // Save the model
        weka.core.SerializationHelper.write("Models/IBK_Model_TS.model", knn);
        System.out.println("Model saved successfully.");

        // Load the model (optional)
        IBk loadedKnn = (IBk) weka.core.SerializationHelper.read("Models/IBK_Model_TS.model");
        System.out.println("Loaded model successfully.");

        // Evaluate the model using training data
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(dataset);
        eval.evaluateModel(loadedKnn, dataset); // Evaluate the model on the same dataset
        long endTime = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;

        // Print evaluation summary
        System.out.println("\nTraining dataset evaluation results:");
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
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
