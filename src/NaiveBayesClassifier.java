package src;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;

public class NaiveBayesClassifier{
    public static void main(String[] args) throws Exception {
        DataSource source1 = new DataSource("Dataset/Titanic1.arff");
        Instances dataset1 = source1.getDataSet();

        dataset1.setClassIndex(0);

        NaiveBayes bayes1 = new NaiveBayes();
        bayes1.buildClassifier(dataset1);

        weka.core.SerializationHelper.write("Models/Naive_Bayes_1.model", bayes1);

        NaiveBayes loadedBayes1 = (NaiveBayes) weka.core.SerializationHelper.read("Models/Naive_Bayes_1.model");

        //Evaluation
        long startTime1 = System.currentTimeMillis(); // Record start time
        Evaluation eval1 = new Evaluation(dataset1);
        eval1.crossValidateModel(loadedBayes1, dataset1, 10, new Random(1));
        long endTime1 = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis1 = endTime1 - startTime1;
        double runtimeSeconds1 = runtimeMillis1 / 1000.0;

        System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval1.areaUnderROC(0));
        System.out.println("kappa = " + eval1.kappa());
        System.out.println("MAE = " + eval1.meanAbsoluteError());
        System.out.println("RMSE = " + eval1.rootMeanSquaredError());
        System.out.println("RAE = " + eval1.relativeAbsoluteError());
        System.out.println("RRSE = " + eval1.rootRelativeSquaredError());
        System.out.println("Error Rate = " + eval1.errorRate());
        System.out.println(eval1.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval1.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds1);

        DataSource source2 = new DataSource("Dataset/Titanic2.arff");
        Instances dataset2 = source2.getDataSet();

        dataset2.setClassIndex(0);

        NaiveBayes bayes2 = new NaiveBayes();
        bayes2.buildClassifier(dataset2);

        weka.core.SerializationHelper.write("Models/Naive_Bayes_2.model", bayes2);

        NaiveBayes loadedBayes2 = (NaiveBayes) weka.core.SerializationHelper.read("Models/Naive_Bayes_2.model");

        //Evaluation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval2 = new Evaluation(dataset2);
        eval2.crossValidateModel(loadedBayes2, dataset2, 10, new Random(1));
        long endTime2 = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis2 = endTime2 - startTime2;
        double runtimeSeconds2 = runtimeMillis2 / 1000.0;

        System.out.println(eval2.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval2.areaUnderROC(0));
        System.out.println("kappa = " + eval2.kappa());
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
