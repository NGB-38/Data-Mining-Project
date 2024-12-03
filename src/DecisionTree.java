package src;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import javax.swing.*;

public class DecisionTree {
    public static void main(String[] args) throws Exception {
        //Loading data
        DataSource source = new DataSource("Dataset/TitanicNom.arff");
        Instances dataset = source.getDataSet();

        //Setting target feature
        dataset.setClassIndex(0);

        //Building model
        J48 tree = new J48();
        tree.buildClassifier(dataset);
//        System.out.println(tree.graph());

        weka.core.SerializationHelper.write("Models/Decision_Tree.model", tree);

        J48 loadedTree = (J48) weka.core.SerializationHelper.read("Models/Decision_Tree.model");

        //Evaluation using cross validation
        long startTime = System.currentTimeMillis(); // Record start time
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(loadedTree, dataset, 10, new Random(1));
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
        System.out.println("Precison = " + eval.precision(0));
        System.out.println("Recall = " + eval.recall(0));
        System.out.println("fMeasure = " + eval.fMeasure(0));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println("Precision (Not survived) = " + eval.precision(0) + ", Precision (Survived) = " + eval.precision(1));
        System.out.println("Recall (Not survived) = " + eval.recall(0) + ", Recall (Survived) = " + eval.recall(1));
        System.out.println("F-Measure (Not survived) = " + eval.fMeasure(0) + ", F-Measure (Survived) = " + eval.fMeasure(1));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds);

        // Visualize the tree in a GUI
        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());

        JFrame frame = new JFrame("J48 Decision Tree");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        frame.getContentPane().add(tv);
        frame.setVisible(true);
    }
}
