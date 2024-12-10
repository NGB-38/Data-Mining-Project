package src;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

public class IBK {
    public static void main(String[] args) throws Exception {
        // Process first dataset
        DataSource source1 = new DataSource("Dataset/Titanic1.arff");
        Instances dataset1 = source1.getDataSet();
        dataset1.setClassIndex(0); // Set the class attribute (e.g., 'Survived')

        IBk knn1 = new IBk();
        knn1.setKNN(30); // Set number of neighbors
        knn1.buildClassifier(dataset1);

        weka.core.SerializationHelper.write("Models/IBk_1.model", knn1);

        IBk loadedKnn1 = (IBk) weka.core.SerializationHelper.read("Models/IBk_1.model");

        // Evaluate first dataset
        long startTime1 = System.currentTimeMillis();
        Evaluation eval1 = new Evaluation(dataset1);
        eval1.crossValidateModel(loadedKnn1, dataset1, 10, new Random(1));
        long endTime1 = System.currentTimeMillis();

        double runtimeSeconds1 = (endTime1 - startTime1) / 1000.0;

        // Print results for dataset1
        System.out.println("=== Results for Dataset 1 ===");
        System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval1.areaUnderROC(0));
        System.out.println("Kappa = " + eval1.kappa());
        System.out.println("MAE = " + eval1.meanAbsoluteError());
        System.out.println("RMSE = " + eval1.rootMeanSquaredError());
        System.out.println("RAE = " + eval1.relativeAbsoluteError());
        System.out.println("RRSE = " + eval1.rootRelativeSquaredError());
        System.out.println("Error Rate = " + eval1.errorRate());
        System.out.println(eval1.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval1.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds1);

        // Process second dataset
        DataSource source2 = new DataSource("Dataset/Titanic2.arff");
        Instances dataset2 = source2.getDataSet();
        dataset2.setClassIndex(0); // Set the class attribute (e.g., 'Survived')

        IBk knn2 = new IBk();
        knn2.setKNN(30); // Set number of neighbors
        knn2.buildClassifier(dataset2);

        weka.core.SerializationHelper.write("Models/IBk_2.model", knn2);

        IBk loadedKnn2 = (IBk) weka.core.SerializationHelper.read("Models/IBk_2.model");

        // Evaluate second dataset
        long startTime2 = System.currentTimeMillis();
        Evaluation eval2 = new Evaluation(dataset2);
        eval2.crossValidateModel(loadedKnn2, dataset2, 10, new Random(1));
        long endTime2 = System.currentTimeMillis();

        double runtimeSeconds2 = (endTime2 - startTime2) / 1000.0;

        // Print results for dataset2
        System.out.println("=== Results for Dataset 2 ===");
        System.out.println(eval2.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval2.areaUnderROC(0));
        System.out.println("Kappa = " + eval2.kappa());
        System.out.println("MAE = " + eval2.meanAbsoluteError());
        System.out.println("RMSE = " + eval2.rootMeanSquaredError());
        System.out.println("RAE = " + eval2.relativeAbsoluteError());
        System.out.println("RRSE = " + eval2.rootRelativeSquaredError());
        System.out.println("Error Rate = " + eval2.errorRate());
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds2);
    }
}
