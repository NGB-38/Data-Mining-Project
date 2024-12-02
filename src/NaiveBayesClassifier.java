package src;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.Random;

public class NaiveBayesClassifier{
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("Dataset/TitanicNom.arff");
        Instances dataset = source.getDataSet();

        dataset.setClassIndex(0);

        NaiveBayes bayes = new NaiveBayes();
        bayes.buildClassifier(dataset);

        weka.core.SerializationHelper.write("Models/Naive_Bayes.model", bayes);

        NaiveBayes loadedBayes = (NaiveBayes) weka.core.SerializationHelper.read("Models/Naive_Bayes.model");

        //Evaluation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(loadedBayes, dataset, 10, new java.util.Random(1));
        long endTime = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;

        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println("AUC = " + eval.areaUnderROC(0));
        System.out.println("kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precision = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("fMeasure = " + eval.fMeasure(0));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
    }
}
