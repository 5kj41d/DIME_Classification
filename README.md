# DIME_Classification
Classifiying archaeological artefacts from the database DIME by Moesgaard Museum.
README will be updated.

Commands:
python3 -m venv pyvenv
pip install -r requirements.txt

Idea and steps: 

    - Sanitize the dataset and investigate different techniques including different image processing techniques or CNN for feature extraction:
        https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
        https://medium.com/@deepanshut041/introduction-to-sift-scale-invariant-feature-transform-65d7f3a72d40 
        https://towardsdatascience.com/exploring-feature-extraction-with-cnns-345125cefc9a 
    This should be seen as a semi supervised learning approach s.t. images are clustered with simular features using either SVMs or K-NN. 
        Goal: To discard non artefacts or images that are bad. 

    The information based on the sanitized dataset: 
        Start classifying all the different kind of artefacts that exists in DIME, based on features and labels.
            Might think using transfer learning or active learning in this case.
        Train a CNN or pre-trained ViT model for this purpose. 
        Random Search for approx. best model. 

        
