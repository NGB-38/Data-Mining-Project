import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;

public class CSV2Arff{
    public static void main(String[]args) throws Exception{
        CSVLoader loader= new CSVLoader();
        loader.setSource(new File("Dataset/Titanic after processing.csv"));
        Instances data= loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("Dataset/Titanic.arff"));
        saver.writeBatch();
    }
}


