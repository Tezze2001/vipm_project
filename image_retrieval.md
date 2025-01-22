TODO:
- FIRST_METHOD: 10-fold stratified CV with 90-10 partition (labelled training) on labelled training.
Steps for each fold:
  - compute centroids for each class as mean of features between images of the same class
  - predict class of validation features using kNN
  - compute accuracy
  
  test: 
    - [x] 1NN with **MEAN and STD for each RGB channels**: Mean accuracy 0.010558, STD accuracy 0.004884
    - [x] 5NN with **MEAN and STD for each RGB channels**: Mean accuracy 0.0095618, STD accuracy 0.0037327
    - [x] 1NN **relu5 alexnet**: Mean accuracy Mean accuracy: 0.020518, STD accuracy: 0.0065093
    - [x] 5NN **relu5 alexnet**: Mean accuracy Mean accuracy: 0.012749, STD accuracy: 0.0029991
    - [x] 1NN **relu7 alexnet**: Mean accuracy 0.016534, STD accuracy: 0.0040986
    - [x] 5NN **relu7 alexnet**: Mean accuracy 0.010757, STD accuracy: 0.0043237


- SECOND_METHOD: 10-fold stratified CV with 90-10 partition (labelled training) on labelled training.
Steps for each fold:
  - predict class of validation features using kNN + standardization
  - compute accuracy
  
  test: 
    - [X] 1NN with hist of HSV: Mean accuracy: 0.30219, STD accuracy: 0.020054 
    - [X] 5NN with hist of HSV: Mean accuracy: Mean accuracy: 0.31673, STD accuracy: 0.019832 
    - [ ] 5NN with MEAN and STD for each RGB channels:
    - [ ] 1NN relu5 alexnet: Mean accuracy Mean accuracy:  0.0039841, STD accuracy: 9.1428e-19
    - [ ] 5NN relu5 alexnet: Mean accuracy Mean accuracy: 0.0039841, STD accuracy: 9.1428e-19
    - [ ] 1NN relu7 alexnet:
    - [ ] 5NN relu7 alexnet:





# Approaches

2 approaches:
 
- first:
    - features from ResNET50
    - compute labels 5NN using as ds small training and qs
- second:
    - features from ResNET50
    - compute unlabelled groups using kmeans
    - for each group, each element is labelled using 5NN from small training and it votes the label for the group, highest classes vote is the label of the group
   

