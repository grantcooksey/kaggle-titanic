# Lessons

This file will be a repository of where I will store things that I have learned 
during this challenge.

* Explore both the train and the test dataset and check for missing data
* One-hot encoding does not handle missing data very well,  I had to hack the training feature set since the test set had a single new value in Parch that did not appear in the training data. Care must be taken to impute missing data or encode a catch all in the data.
* Make sure you have fully explorered and understood the dataset.
* Discrete does **not** mean catagorical!
    
# Specific

* I treated SibSp and Parch as non linear discrete values but this may not be an appropriate choice.
* Take a look at an interaction feature between age and fare
* Check out decision trees as the model.  That would probably handle the data better.