package src;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;
import java.util.Random;

public class DecisionTree {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("Dataset/Titanic.arff");
        Instances dataset = source.getDataSet();

        NumericToNominal numToNom = new NumericToNominal();
        numToNom.setAttributeIndices("first");
        numToNom.setInputFormat(dataset);

        Instances newData = Filter.useFilter(dataset, numToNom);
        newData.setClassIndex(0);

        J48 tree = new J48();
        tree.buildClassifier(newData);
        System.out.println(tree.graph());

        File modelDir = new File("Models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }

        weka.core.SerializationHelper.write("Models/Decision_Tree.model", tree);

        J48 loadedTree = (J48) weka.core.SerializationHelper.read("Models/Decision_Tree.model");

        //Evaluation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(newData);
        eval.crossValidateModel(loadedTree, newData, 10, new java.util.Random(1));
        long endTime = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis = endTime - startTime;
        double runtimeSeconds = runtimeMillis / 1000.0;

        System.out.println(eval.toSummaryString("Evaluation results:\n", false));
        System.out.println("Correct % = "+eval.pctCorrect());
        System.out.println("Incorrect % = "+eval.pctIncorrect());
        System.out.println("AUC = " + eval.areaUnderROC(0));
        System.out.println("kappa = " + eval.kappa());
        System.out.println("MAE = " + eval.meanAbsoluteError());
        System.out.println("RMSE = " + eval.rootMeanSquaredError());
        System.out.println("RAE = " + eval.relativeAbsoluteError());
        System.out.println("RRSE = " + eval.rootRelativeSquaredError());
        System.out.println("Precison = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("fMeasure = " + eval.fMeasure(0));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);
    }
}
