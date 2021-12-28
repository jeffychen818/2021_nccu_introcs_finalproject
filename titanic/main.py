# %%
import numpy as np
from numpy import matrixlib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test.csv')
total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
total_data = total_data.drop(columns=['Cabin', 'PassengerId'])

# %%
# feature enginnering
total_data['Title'] = total_data.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0]
title_age = total_data[['Sex', 'Title', 'Age']]
total_data = total_data.drop(columns=['Name'])

# fill in the missing age
title_count = title_age.groupby(['Sex', 'Title']).count()
title_age = title_age.groupby(['Title', 'Sex'])['Age'].mean().unstack()

# find the title of NaN Age
def check_title_age():
    missing_list = []
    for index in total_data.index:
        if (pd.isna(total_data.iloc[index]['Age'])) == True:
            missing_list.append(total_data.iloc[index]['Title'])
    missing_list = pd.Series(missing_list)
    missing_list = missing_list.value_counts().reset_index()
    missing_list.columns = ['title', 'count']
    for index in missing_list.index:
        print("{} has missed {} age value".format(missing_list.iloc[index]['title'],
                                                  missing_list.iloc[index]['count']))
# check_title_age()

# fill in age value
def fillin_missing_age():
    def title_data():
        total_data['Title'] = total_data.Title.replace(['Miss', 'Mlle', 'Mme', 'Ms', 'Lady'], 'Mrs', regex=True)
        re_title_age = total_data[['Title', 'Age']].dropna().groupby('Title', as_index=False)['Age'].median()
        return re_title_age
    re_title_age = title_data()
    for index in total_data[total_data['Age'].isnull()].index:
        title = total_data.iloc[index]['Title']
        age = re_title_age.loc[re_title_age['Title'] == str(title),
                               'Age']
        total_data.loc[index,'Age'] = int(age)
fillin_missing_age()

# fill in missing fare
def fillin_missing_fare():
    fare_data = total_data[['Pclass', 'Embarked', 'Fare']].groupby(['Pclass', 'Embarked'])['Fare'].mean().unstack()
    fare = 0
    for index in total_data[total_data.Fare.isnull()].index:
        family = total_data.iloc[index]['SibSp'] + total_data.iloc[index]['Parch']
        ticket = total_data.iloc[index]['Ticket']
        if family == 0:
            person_class = total_data.iloc[index]['Pclass']
            person_embarked = total_data.iloc[index]['Embarked']
            fare = fare_data.loc[person_class, person_embarked]
        else:
             fare = total_data[(total_data['Ticket']==ticket) &
                             total_data['Fare'].isnull() == False]['Fare']
        total_data.loc[index, 'Fare'] = fare
fillin_missing_fare()

# %%
# count family size
def family_size():
    for index in total_data.index:
        size = 0
        family = total_data.iloc[index]['SibSp'] + total_data.iloc[index]['Parch']
        if (family == 0):
            size = 0
        elif (1 < family < 3):
            size = 2
        else:
            size = 1
        total_data.loc[total_data.index==index, 'Family'] = int(size)

family_size()

def encoded_age():
    for index in total_data.index:
        mark = 0
        age = total_data.iloc[index]['Age']
        if (age < 17):
            mark = 0
        elif (17<= age <= 33):
            mark = 1
        else:
            mark = 2
        total_data.loc[total_data.index==index, 'Age'] = int(mark)
    encoded_age = pd.get_dummies(total_data['Age'], prefix='Age')
    data = pd.concat([total_data, encoded_age], axis=1)
    return data

total_data = encoded_age()


# 1 => female, 0 => male
def encoding_sex():
    total_data['Sex'] = total_data['Sex'].map({
        'female': 1,
        'male': 0
    }).astype(int)

encoding_sex()

# encoding embarked data
def encoding_embarked():
    encoded_embarked = pd.get_dummies(total_data['Embarked'], prefix='Embared')
    data = pd.concat([total_data, encoded_embarked], axis=1)
    return data

total_data = encoding_embarked()

# encoding Pclass data
def encoding_pclass():
    encoded_pclass = pd.get_dummies(total_data['Pclass'], prefix='Pclass')
    data = pd.concat([total_data, encoded_pclass], axis=1)
    return data

total_data = encoding_pclass()
# encloding Family
def encoding_family():
    encoded_family = pd.get_dummies(total_data['Family'], prefix='Family')
    data = pd.concat([total_data, encoded_family], axis=1)
    return data

total_data = encoding_family()

total_data = total_data.drop(columns=[
    'Embarked', 'Ticket', 'Title', 'SibSp', 'Parch', \
        'Pclass', 'Family'])

# %% preview coordination
def view_coordination():
    coordination = total_data.corr()
    coordination = coordination.loc['Survived',:].sort_values()[:-1]
    coordination_df = pd.DataFrame({'Survived': coordination})
    print(coordination_df)
# view_coordination()


# %% machine learning part

# load data
train = total_data[total_data.Survived.notnull()]
test_feature = total_data[total_data.Survived.isnull()]

test_feature = test_feature.drop(columns=['Survived'])
# create features & labels
def create_feature_labels(data):
    train_feature = data[data.columns.drop('Survived')]
    train_label = data['Survived']
    return train_feature, train_label

# train features & labels
feature, label = create_feature_labels(train)

# test features & labels
test_data = pd.read_csv('./gender_submission.csv')
test_label = test_data['Survived']

'''
--------------------
decision model
--------------------
'''
def testing_decisiontree_model(x_train, x_test, y_train, y_test, rate):
    depth_range = np.arange(3,15)
    accuracy = np.array([])
    for i in depth_range:
        dct = DecisionTreeClassifier(max_depth=i)
        adb = AdaBoostClassifier(dct, n_estimators=600, learning_rate=rate)
        adb.fit(x_train, y_train)
        adb.predict(x_test)
        score = adb.score(x_test, y_test)
        accuracy = np.append(accuracy, score)
    plt.plot(depth_range, accuracy)
    plt.show()

testing_decisiontree_model(feature, test_feature, label, test_label, 0.1)

# def adaboost_model(x_train, x_test, y_train, y_test, learningrate):
#     dct = DecisionTreeClassifier(max_depth=12)
#     adb = AdaBoostClassifier(dct, n_estimators=450, learning_rate=learningrate)
#     adb.fit(x_train, y_train)
#     adb.predict(x_test)
#     score = adb.score(x_test, y_test)
#     print(f"Accuracy: {score}")

# adaboost_model(feature, test_feature, label, test_label, 0.1)


'''
--------------------
randon forest model
--------------------
'''

def randomforest_model(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=1000, oob_score=True)
    rfc.fit(x_train, y_train)
    rfc.predict(x_test)
    score = rfc.score(x_test, y_test)
    print(f"Accuracy: {score}")

# randomforest_model(feature, test_feature, label, test_label)
    
    