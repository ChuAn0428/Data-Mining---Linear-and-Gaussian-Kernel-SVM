We will use scikit-learn SVM tool for classification. We will use “Congressional Voting Records Data Set” available from UCI Machine Learning Repository. This is a classification dataset with two classes, namely democratic, and repub- lican. It has 16 features, all of them binary (y value represents yes, n value represents no, and ? value represents withdrawal from voting).

a.	Convert this dataset into numeric by converting y to 1, n to -1 and ? to 0.

b.	Break the dataset into 4 folds with approximately similar ratio of the classes (republi- can/democratic) in each fold. Use 1 fold for parameter tuning only. For the remaining three folds, report average 3-fold classification accuracy (along with standard deviation) for Linear SVM with soft-margin classifier.

c.	Now use SVM with Gaussian Kernel for the same task.
