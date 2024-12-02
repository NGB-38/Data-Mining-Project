package src;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class NaiveBayes_TrainingSet {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("Dataset/Titanic.arff");
        Instances dataset = source.getDataSet();

        // Convert 'Survived' attribute to nominal if it's numeric
        NumericToNominal numToNom = new NumericToNominal();
        numToNom.setAttributeIndices("1"); // First attribute (Survived)
        numToNom.setInputFormat(dataset);
        Instances newData = Filter.useFilter(dataset, numToNom);

        // Set the class attribute (target variable)
        newData.setClassIndex(0); // 'Survived' is the first attribute

        // Build Naive Bayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(newData);

        // Save the model
        File modelDir = new File("Models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
        weka.core.SerializationHelper.write("Models/NaiveBayes_Model_TS.model", nb);
        System.out.println("Model saved successfully.");

        // Load the model (optional)
        NaiveBayes loadedNb = (NaiveBayes) weka.core.SerializationHelper.read("Models/NaiveBayes_Model_TS.model");
        System.out.println("Loaded model successfully.");

        // Evaluate the model using training data
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(newData);
        eval.evaluateModel(nb, newData); // Evaluate the model on the same dataset
        System.out.println("\nTraining dataset evaluation results:");
        System.out.println(eval.toSummaryString("Evaluation results:\n", false));

        long endTime = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;

        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Incorrect % = " + eval.pctIncorrect());
        System.out.println("AUC = " + eval.areaUnderROC(0));
        System.out.println("Kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precision (index 0) = " + eval.precision(0) + ", Precision (index 1) = " + eval.precision(1));
        System.out.println("Recall (index 0) = " + eval.recall(0) + ", Recall (index 1) = " + eval.recall(1));
        System.out.println("F-Measure (index 0) = " + eval.fMeasure(0) + ", F-Measure (index 1) = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
    }
}
