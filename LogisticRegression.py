import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import textstat
from textstat import flesch_reading_ease
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
#%%FUNCTIONS
def deriveFeatures(dataframe): # Call this to derive features :)
	#convert dates to datetime format
	dataframe[['created_at', 'deadline', 'launched_at']] = dataframe[['created_at', 'deadline', 'launched_at']].apply(pd.to_datetime,unit='s')

	#derive duration
	dataframe['duration'] = (dataframe['deadline'] - dataframe['launched_at'])

	#derive word count for name & blurb
	dataframe['word_count'] = get_word_count(dataframe['blurb'])
	dataframe['name_count'] = get_word_count(dataframe['name'])	

	#calculate same day launches
	get_same_day_launced(dataframe)



def get_word_count(blurb_list):
    word_list =[]
    for blurb_index in blurb_list:
        word_list.append(len(str(blurb_index).split()))
    return word_list


def get_same_day_launced(dataframe): #calculate how many prjects were launched on the same day
	for key in ['launched_at', 'created_at', 'deadline']: 
		dataframe[key] = dataframe[key].dt.date
	
	possibleDates = dataframe['launched_at'].unique()
	dataframe['same_day_projects'] = 0
	for date in possibleDates:
		sameDayProj = dataframe.loc[dataframe['launched_at']==date]
		dataframe['same_day_projects'].loc[dataframe['launched_at']==date] = len(sameDayProj)	


#%%WORDCLOUD
Xtest=pd.read_csv("D:\\USSER\\Downloads\\KS_test_data.csv",delimiter=';')
train=pd.read_csv("D:\\USSER\\Downloads\\KS_train_data.csv",delimiter=',')
read=[]
for i in train['blurb']:
    read.append(textstat.flesch_reading_ease(str(i)))
train['readability']=read

readtest=[]
for i in Xtest['blurb']:
    readtest.append(textstat.flesch_reading_ease(str(i)))
Xtest['readability']=readtest
deriveFeatures(train)
deriveFeatures(Xtest)
Xtest=Xtest[['goal','staff_pick','category','country','project_id','readability','word_count','name_count','same_day_projects']]
Xtest=Xtest.set_index('project_id')
Xtrain=train[['goal','staff_pick','project_id','category','country','readability','word_count','name_count','same_day_projects']]
Xtrain=Xtrain.set_index('project_id')
Ytrain=train[['funded','project_id']]
Ytrain=Ytrain.set_index('project_id')
le=LabelEncoder()
Xtest['category']=le.fit_transform(Xtest['category'])
Xtrain['category']=le.fit_transform(Xtrain['category'])
Xtest['country']=le.fit_transform(Xtest['country'])
Xtrain['country']=le.fit_transform(Xtrain['country'])

#splitting the train set
Xtrainmodel, Xtestmodel, ytrainmodel, ytestmodel = train_test_split(Xtrain, Ytrain, test_size=0.3) 
logreg=LogisticRegression()


#BEST HYPERPARAMETERS
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(Xtrain,Ytrain)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

#APPLY MODEL WITH BEST HYPERPARAMETERS
logreg2=LogisticRegression(C=1,penalty="l2")
logreg2.fit(Xtrainmodel,ytrainmodel)

scores2=cross_val_score(logreg2,Xtestmodel,ytestmodel,cv=30)

scores2 = pd.Series(scores2)
scores2.min(), scores2.mean(), scores2.max()

