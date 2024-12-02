package src;

import weka.classifiers.lazy.IBk;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class IBK {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Dataset/TitanicNom.arff");
        Instances dataset = source.getDataSet();

        // Set the class attribute (target variable)
        dataset.setClassIndex(0); // 'Survived' is the first attribute

        // Build IBK classifier
        IBk knn = new IBk();
        knn.setKNN(30); // Set the number of neighbors (k) (root mean square of instances)
        knn.buildClassifier(dataset);

        // Save the model
        weka.core.SerializationHelper.write("Models/IBK_Model_CV.model", knn);
        System.out.println("Model saved successfully.");

        // Load the model (optional)
        IBk loadedKnn = (IBk) weka.core.SerializationHelper.read("Models/IBK_Model_CV.model");
        System.out.println("Loaded model successfully.");

        // Evaluate the model
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(loadedKnn, dataset, 10, new Random(1)); // 10-fold cross-validation
        long endTime = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;

        // Print evaluation summary
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval.areaUnderROC(0));
        System.out.println("Kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precision (Not survived) = " + eval.precision(0) + ", Precision (Survived) = " + eval.precision(1));
        System.out.println("Recall (Not survived) = " + eval.recall(0) + ", Recall (Survived) = " + eval.recall(1));
        System.out.println("F-Measure (Not survived) = " + eval.fMeasure(0) + ", F-Measure (Survived) = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
    }
}
