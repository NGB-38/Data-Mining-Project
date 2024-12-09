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
        DataSource source1 = new DataSource("Dataset/Titanic1.arff");
        Instances dataset1 = source1.getDataSet();

        //Setting target feature
        dataset1.setClassIndex(0);

        //Building model
        J48 tree1 = new J48();
        tree1.buildClassifier(dataset1);
//        System.out.println(tree.graph());

        weka.core.SerializationHelper.write("Models/Decision_Tree_1.model", tree1);

        J48 loadedTree1 = (J48) weka.core.SerializationHelper.read("Models/Decision_Tree.model");

        //Evaluation using cross validation
        long startTime1 = System.currentTimeMillis(); // Record start time
        Evaluation eval1 = new Evaluation(dataset1);
        eval1.crossValidateModel(loadedTree1, dataset1, 10, new Random(1));
        long endTime1 = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis1 = endTime1 - startTime1;
        double runtimeSeconds1 = runtimeMillis1 / 1000.0;

        System.out.println(eval1.toSummaryString("Case 1 evaluation results:\n", false));
        System.out.println("AUC = " + eval1.areaUnderROC(0));
        System.out.println("kappa = " + eval1.kappa());
        System.out.println("MAE = " + eval1.meanAbsoluteError());
        System.out.println("RMSE = " + eval1.rootMeanSquaredError());
        System.out.println("RAE = " + eval1.relativeAbsoluteError());
        System.out.println("RRSE = " + eval1.rootRelativeSquaredError());
        System.out.println("fMeasure = " + eval1.fMeasure(0));
        System.out.println("Error Rate = " + eval1.errorRate());
        System.out.println(eval1.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval1.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds1);

//        // Visualize the tree in a GUI
//        TreeVisualizer tv = new TreeVisualizer(null, tree1.graph(), new PlaceNode2());
//
//        JFrame frame = new JFrame("J48 Decision Tree case 1");
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setSize(800, 600);
//        frame.getContentPane().add(tv);
//        frame.setVisible(true);


        //Loading data
        DataSource source2 = new DataSource("Dataset/Titanic2.arff");
        Instances dataset2 = source2.getDataSet();

        //Setting target feature
        dataset2.setClassIndex(0);

        //Building model
        J48 tree2 = new J48();
        tree2.buildClassifier(dataset2);

        weka.core.SerializationHelper.write("Models/Decision_Tree_2.model", tree2);

        J48 loadedTree2 = (J48) weka.core.SerializationHelper.read("Models/Decision_Tree_2.model");

        //Evaluation using cross validation
        long startTime2 = System.currentTimeMillis(); // Record start time
        Evaluation eval2 = new Evaluation(dataset2);
        eval2.crossValidateModel(loadedTree2, dataset2, 10, new Random(1));
        long endTime2 = System.currentTimeMillis();

        // Calculate runtime
        long runtimeMillis2 = endTime2 - startTime2;
        double runtimeSeconds2 = runtimeMillis2 / 1000.0;

        System.out.println(eval2.toSummaryString("Case 2 evaluation results:\n", false));
        System.out.println("AUC = " + eval2.areaUnderROC(0));
        System.out.println("kappa = " + eval2.kappa());
        System.out.println("MAE = " + eval2.meanAbsoluteError());
        System.out.println("RMSE = " + eval2.rootMeanSquaredError());
        System.out.println("RAE = " + eval2.relativeAbsoluteError());
        System.out.println("RRSE = " + eval2.rootRelativeSquaredError());
        System.out.println("fMeasure = " + eval2.fMeasure(0));
        System.out.println("Error Rate = " + eval2.errorRate());
        System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
        System.out.println(eval2.toMatrixString("\n=== Overall Confusion Matrix ===\n"));
        System.out.println("Runtime (seconds): " + runtimeSeconds2);

//        // Visualize the tree in a GUI
//        TreeVisualizer tv2 = new TreeVisualizer(null, tree2.graph(), new PlaceNode2());
//
//        JFrame frame2 = new JFrame("J48 Decision Tree case 2");
//        frame2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame2.setSize(800, 600);
//        frame2.getContentPane().add(tv2);
//        frame2.setVisible(true);
    }
}
