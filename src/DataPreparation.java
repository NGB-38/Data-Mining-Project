package src;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.io.File;

public class DataPreparation {
    public static void main (String[] args) throws Exception{
        //convert numeric target attribute into nominal
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("Dataset/Titanic.arff");
        Instances dataset = source.getDataSet();

        NumericToNominal numToNom = new NumericToNominal();
        numToNom.setAttributeIndices("first");
        numToNom.setInputFormat(dataset);

        Instances newData = Filter.useFilter(dataset, numToNom);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(newData);
        saver.setFile(new File("Dataset/TitanicNom.arff"));
        saver.writeBatch();

        // Create the directory if it doesn't exist
        File modelDir = new File("Models");
        if (!modelDir.exists()) {
            modelDir.mkdirs();
        }
    }
}
