'''
The toolbox package contains a set of scripts and Python modules to
run the tracking pipeline -- mostly without ilastik.

The important **scripts** are:

* `toolbox.create_label_image_from_raw`: after running *Pixel Classification* on a dataset,
 this script takes the probability maps and applies thresholding to get a segmentation,
 using the settings of an *ilastik Tracking* project.
* `toolbox.train_transition_classifier`: given a ground truth (as one .h5 file per frame) and
 corresponding raw data, it trains a random forest classifier to predict the quality of a transition.
* `toolbox.hypotheses_graph_to_json`: creates a JSON hypotheses graph given raw images, 
 their segmentations, and trained classifiers (can use a transition classifier from above).

The important **modules** are:

* `toolbox.ilastik_project_options`
* `toolbox.random_forest_classifier`
* `toolbox.traxelstore`
* `toolbox.hypothesesgraph`
* `toolbox.divisionfeatures`
* `toolbox.progressbar`

The traxelstore and some of the scripts use the `toolbox.pluginsystem` and 
[`yapsy`](http://yapsy.sourceforge.net/) to make the pipeline easy to extend.

'''
