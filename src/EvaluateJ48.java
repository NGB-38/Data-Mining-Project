package src;

import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class EvaluateJ48 {
    public static void main(String args[]) {
        try {
            // Load training data
            DataSource source = new DataSource("C:\\Users\\admin\\Desktop\\Data-Mining-Project\\Dataset\\weather.nominal.arff");
            Instances dataset = source.getDataSet();

            // Set the class index to the last attribute
            dataset.setClassIndex(dataset.numAttributes() - 1);

            // Create and build the custom J48 classifier
            J48 tree = new J48();
            tree.buildClassifier(dataset);

            // Initialize evaluation with 10-fold cross-validation
            Evaluation eval = new Evaluation(dataset);
            Random rand = new Random(1); // Seed for reproducibility
            int folds = 10;

            // Perform cross-validation
            eval.crossValidateModel(tree, dataset, folds, rand);

            // Print evaluation results
            System.out.println(eval.toSummaryString("Evaluation results:\n", false));
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("Kappa = " + eval.kappa());
            System.out.println("Mean Absolute Error = " + eval.meanAbsoluteError());
            System.out.println("Root Mean Square Error = " + eval.rootMeanSquaredError());
            System.out.println("Relative Absolute Error = " + eval.relativeAbsoluteError());
            System.out.println("Root Relative Square Error = " + eval.rootRelativeSquaredError());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("F-Measure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));

        } catch (Exception e) {
            // Handle exceptions gracefully
            System.err.println("An error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
