import multiprocessing
from sklearn.preprocessing import normalize
from spike_detection.mnf_spike_detection import parallel_nmf_consensus_clustering
from preprocessing.pipeline import parallel_preprocessing
from loader.loader import read_h5_file

FILEPATH = "../../../Data/"
FILENAME = "Sz2.h5"

if __name__ == "__main__":
    data = read_h5_file(FILEPATH + FILENAME)
    preprocessed = parallel_preprocessing(data)
    multiprocessing.freeze_support()

    # Load and process your data here (remove load_iris() this is just an example)
    # data = load_iris()
    X = preprocessed

    # If your data set has labels and you want to see if NMF can reproduce them in
    # an unsupervised manner you can input them here, otherwise set to "None"
    labels = None

    # Here you can add any preprocessing steps but make sure your data are properly
    # normalized or scaled for NMF (X should be non-negative)
    data_matrix = normalize(X)

    # If you want to find the optimal k number of clusters you need to first specify
    # a range for k. A larger range will require more computational resources
    # If you are finding it takes too long to run on your local machine you can try running
    # on the UBELIX cluster (see my documentation on getting set up on HPC)
    k_min = 2
    k_max = 5

    # Lastly, you should specify the number of runs per rank.  This is important for estabilishing
    # the stability of your solution per rank.  While not recursive, this process ensures that
    # there are enough replicates to claim stability.  Default = 100
    runs_per_rank = 100

    # Run the NMF consensus clustering
    experiment_dir = parallel_nmf_consensus_clustering(
        data_matrix, (k_min, k_max), runs_per_rank, target_clusters=labels
    )

    # Print a confirmation that the results have been saved in the appropriate directory
    print(f"Results saved in directory: {experiment_dir}")
