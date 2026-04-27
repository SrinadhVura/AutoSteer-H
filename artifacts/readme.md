* This directory contains the raw outptut json files of all the decoding strategies during evaluation.
* The `grid_search_results.csv` has the metrics obtained from all the tested combinations of hyperparameters `alpha` and prober threshold.
* `steering_vector_layer_19.pt` is the stored steering global steering vector.
* `layer_19_features.pt` contains stored feature maps of the layer 19 across *factual* and *hallucinated* data observations.
* `layers_19_labels.pt` contains labels for the saved feature maps as `0` being factual and `1` being hallucinated. 
