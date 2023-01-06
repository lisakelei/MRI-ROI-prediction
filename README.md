# MRI-ROI-prediction
TensorFlow 2.0 implementation of a model predicting region of interest in MRI localizers.

Link to paper:

The model can be trained to prediction the left-right boundries or top-bottom boundries of an ROI for a stack of hip or abdomen MRI localizer. 
Two instances of the same model is trained independently for the left-right and top-bottom pairs of boundries.
The input to the model is a stack of 512x512 2D 1-channel grey images, and the output of the model is two scalars indicating the coordinates within the 512 pixels. 
