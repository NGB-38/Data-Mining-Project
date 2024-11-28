package src;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Clustering {
    public static void main(String args[]) throws Exception {
        // Load data
        String dataset = "Dataset/Titanic.arff";
        DataSource source = new DataSource(dataset);
        Instances data = source.getDataSet();

        // Create and configure the k-means clustering model
        SimpleKMeans model = new SimpleKMeans();
        model.setNumClusters(2); // Set the number of clusters
        model.buildClusterer(data);

        // Print cluster information
        System.out.println(model);

        // Evaluate the clusters using the same dataset
        ClusterEvaluation clsEval = new ClusterEvaluation();
        clsEval.setClusterer(model);
        clsEval.evaluateClusterer(data);

        // Print the number of clusters and other evaluation metrics
        System.out.println("# of clusters: " + clsEval.getNumClusters());
        System.out.println("Cluster Evaluation Results: " + clsEval.clusterResultsToString());
    }
}
