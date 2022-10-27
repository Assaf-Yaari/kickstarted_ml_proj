import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


## inputs are a trainging Dataframe after transformation 

def mean_sd(training_scores):
    return np.mean(training_scores, axis=1) 

### validation-curve 
def validation_curve(cleadDF , target):
    krange = range(1,15,1)

    train_score, test_score = validation_curve(KNeighborsClassifier(), cleadDF, target,
                                        param_name = "n_neighbors",
                                        param_range = krange,
                                        cv = 5, scoring = "accuracy")
                        
    # Calculating mean of training score
    train_score_m  = mean_sd(train_score)
    # Calculating mean of testing score
    test_score_m = mean_sd(test_score)
    
    # Plot mean accuracy scores for training and testing scores
    plt.plot(krange, train_score_m,
        label = "Training Score", color = 'b')
    plt.plot(krange, test_score_m,
    label = "Cross Validation Score", color = 'r')
    
    plt.xlabel("Number of Neighbours")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

# learning curve 
def learning_curve(cleadDF, target):
    sizes, training_scores, testing_scores = learning_curve(KNeighborsClassifier(), cleadDF, target, 
                                                            cv=5, scoring='accuracy')
    # Mean and Standard Deviation of training scores
    train_score_m  = mean_sd(training_scores)

    # Mean and Standard Deviation of testing scores
    test_score_m = mean_sd(testing_scores)

    # dotted blue line is for training scores and green line is for cross-validation score
    plt.plot(sizes, train_score_m, '--', color="b",  label="Training score")
    plt.plot(sizes, test_score_m, color="g", label="Cross-validation score")
    
    # Drawing plot
    plt.title("LEARNING CURVE FOR KNN Classifier")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


#BEST HYPERPARAMETERS
def outcome(cleadDF , target):
    k_range = list(range(1,15,1))
    param_grid = dict(n_neighbors=k_range)
    bayesModel = KNeighborsClassifier()
    grid = GridSearchCV(bayesModel, param_grid, cv=5, scoring='accuracy')
    grid.fit(cleadDF,target)
    print("tuned hpyerparameters :(best parameters) ",grid.best_params_)
    print("accuracy :",grid.best_score_)

#APPLY MODEL WITH BEST HYPERPARAMETERS
def train_with_best_param(cleadDF ,target):
    Xtrain, Xtest, ytrain, ytest = train_test_split(cleadDF, target, test_size=0.3) # Train:Test = 70%:30%
    bayesModel = KNeighborsClassifier(n_neighbors = 13)
    bayesModel.fit(Xtrain,ytrain)
    y_model = bayesModel.predict(Xtest)
    print(accuracy_score(ytest, y_model))



    
    


