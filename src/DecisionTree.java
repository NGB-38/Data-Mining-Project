package src;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

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

//        weka.core.SerializationHelper.write("Models/Decision_Tree.model", tree);

        J48 loadedTree = (J48) weka.core.SerializationHelper.read("Models/Decision_Tree.model");
    }
}
