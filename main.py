from os import write
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Generic function for making a classification model and accessing performance:


def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors], data[outcome])
    predictions = model.predict(data[predictors])
    # st.line_chart(predictions)
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    st.write(f'Accuracy : `{round(accuracy*100,2)}%`')


plt.style.use('dark_background')
plt.rcParams.update({'font.size': 14})
matches = pd.read_csv('matches.csv')


predictor_var = ['team1', 'team2', 'venue',
                 'toss_winner', 'city', 'toss_decision']

teamNames = ['Mumbai Indians', 'Kolkata Knight Riders', 'Royal Challengers Bangalore', 'Deccan Chargers', 'Chennai Super Kings',
             'Rajasthan Royals', 'Delhi Daredevils', 'Gujarat Lions', 'Kings XI Punjab',
             'Sunrisers Hyderabad', 'Rising Pune Supergiants', 'Kochi Tuskers Kerala', 'Pune Warriors']
teamSymbols = ['MI', 'KKR', 'RCB', 'DC', 'CSK', 'RR',
               'DD', 'GL', 'KXIP', 'SRH', 'RPS', 'KTK', 'PW']
encode = {'team1': {'MI': 1, 'KKR': 2, 'RCB': 3, 'DC': 4, 'CSK': 5, 'RR': 6, 'DD': 7, 'GL': 8, 'KXIP': 9, 'SRH': 10, 'RPS': 11, 'KTK': 12, 'PW': 13},
          'team2': {'MI': 1, 'KKR': 2, 'RCB': 3, 'DC': 4, 'CSK': 5, 'RR': 6, 'DD': 7, 'GL': 8, 'KXIP': 9, 'SRH': 10, 'RPS': 11, 'KTK': 12, 'PW': 13},
          'toss_winner': {'MI': 1, 'KKR': 2, 'RCB': 3, 'DC': 4, 'CSK': 5, 'RR': 6, 'DD': 7, 'GL': 8, 'KXIP': 9, 'SRH': 10, 'RPS': 11, 'KTK': 12, 'PW': 13},
          'winner': {'MI': 1, 'KKR': 2, 'RCB': 3, 'DC': 4, 'CSK': 5, 'RR': 6, 'DD': 7, 'GL': 8, 'KXIP': 9, 'SRH': 10, 'RPS': 11, 'KTK': 12, 'PW': 13, 'Draw': 14}}


def introduction():
    intro = f"""
    # Introduction
    - The prediction tool will go through `DataFrame` of players of many teams and will come up with the team which has the **better chance of winning statistically**.
    This is the combination of cricket plus data analysis gives the predictions for the match.

    # Team Members
    - Adarsh M             (1MV18CS005)
    - Ashutosh Shukla      (1MV18CS014)
    - Brinda Kaushik B A   (1MV18CS020)
    - Jai Supreeth         (1MV18CS045)
    - Nithyashree N S      (1MV18CS063)
    - Sanjay A             (1MV18CS089)
    """
    st.write(intro)


def load_data():
    st.write('## Loading `DataFrame` from `matches.csv`')
    st.code('''matches=pd.read_csv('matches.csv')
    ''')
    matches
    st.write('### Replacing all `nan` values with Draw in the **Winner** column.')
    st.code(
        '''matches.loc[pd.isnull(matches['winner']), ['id', 'season', 'winner']]''')
    matches.loc[pd.isnull(matches['winner']), ['id', 'season', 'winner']]
    matches['winner'].fillna('Draw', inplace=True)
    st.code('''
matches[pd.isnull(matches['winner'])]
matches['winner'].fillna('Draw', inplace=True)
    ''')
    matches.loc[matches['winner'] == 'Draw', ['id', 'season', 'winner']]


def clean_encode():
    st.header('Cleaning and Encoding `DataFrame`')
    st.write('### Replacing all the team names with a unique symbol.')
    matches[['team1', 'team2']]
    matches.replace(teamNames, teamSymbols, inplace=True)
    st.code('''matches.replace(teamNames, teamSymbols, inplace=True)''')
    matches[['team1', 'team2']]
    st.write('### Encoding the unique team symbols.')
    matches.replace(encode, inplace=True)
    st.code('''matches.replace(encode, inplace=True)''')
    matches[['team1', 'team2']]


def pda():
    st.header('Preliminary Analysis of the `Dataframe`')
    st.write('Describing the `Dataframe`')
    st.dataframe(matches.describe())
    st.write('We maintain a dictionary for future reference mapping of teams.')
    dicVal = encode['winner']
    st.json(dicVal)
    df = matches[['team1', 'team2', 'city',
                  'toss_decision', 'toss_winner', 'venue', 'winner']]
    st.write('We Find some stats on the *match winners and toss winners*')
    col1, col2 = st.beta_columns(2)
    with col1:
        st.write('No of `Toss winners` by each team:')
        st.dataframe(df['toss_winner'].value_counts())
        st.write('Chart:')
        st.bar_chart(df['toss_winner'].value_counts())
    with col2:
        st.write('No of `Match winners` by each team')
        st.dataframe(df['winner'].value_counts())
        st.write('Chart:')
        st.bar_chart(df['winner'].value_counts())
    st.write(
        'As we can observe `Mumbai` has won most toss and also most matches so far.')
    return df, dicVal


def building_model(df):
    st.header('Building Predictive Model')
    st.write('Converting categorical data to numerical data')
    df
    var_mod = ['city', 'toss_decision', 'venue']
    le = LabelEncoder()  # using SKlearn ML kit
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    st.code('''var_mod = ['city', 'toss_decision', 'venue']
le = LabelEncoder() #using SKlearn ML kit
for i in var_mod:
    df[i] = le.fit_transform(df[i])''')
    df
    st.write('### Importing few ML algorithms from `SCI-KIT`')
    st.code('''# Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors], data[outcome])
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions, data[outcome])''')

    st.write('### Logistic Regression')
    outcome_var = ['winner']
    model = LogisticRegression()
    classification_model(model, df, predictor_var, outcome_var)
    st.write('### Gaussian Naive')
    model = GaussianNB()
    classification_model(model, df, predictor_var, outcome_var)
    st.write('### KNN Algorithm')
    model = KNeighborsClassifier(n_neighbors=3)
    classification_model(model, df, predictor_var, outcome_var)
    st.write('### SVM classification')
    model = svm.SVC(kernel='rbf', C=1, gamma=1)
    classification_model(model, df, predictor_var, outcome_var)
    # st.write('### Gradient Boost Classifier')
    # model = GradientBoostingClassifier(
    #     n_estimators=100, learning_rate=0.2, max_depth=8, random_state=0)
    # classification_model(model, df, predictor_var, outcome_var)
    st.write('### Decision Tree Classifier')
    model = tree.DecisionTreeClassifier(criterion='gini')
    classification_model(model, df, predictor_var, outcome_var)
    st.write('### Random forest classifier')
    model = RandomForestClassifier(n_estimators=100)
    classification_model(model, df, predictor_var, outcome_var)
    return model


def testing(model, dicVal):
    st.header('Applying Random forest Model on `Test Data` ')
    st.write('Importing `test.csv`')
    test = pd.read_csv("test.csv")
    test
    test = test.drop(["date", "winner"], axis=1, inplace=False)
    test.replace(encode, inplace=True)
    st.write('Dropping the `winner` and `date` column and encoding')
    st.code('''
    test = test.drop(["date", "winner"], axis=1, inplace=False)
    test.replace(encode, inplace=True)
    ''')
    test
    st.subheader('Random forest Model Prediction:')
    out = model.predict(test)
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.write('Prediction:')
        out
    with col2:
        st.write('Corresponding teams:')
        predicted_winner = []
        for i in out:
            predicted_winner.append(
                list(dicVal.keys())[list(dicVal.values()).index(i)])
        st.write(predicted_winner)
    with col3:
        st.write('Team Encode Disctionary')
        dicVal
    test = pd.read_csv("test.csv")
    ctr = 0
    k = 0
    total = len(test['winner'])
    for i in test['winner']:
        if i == predicted_winner[k]:
            ctr = ctr+1
        k = k+1
    st.write(
        f'`{ctr}` out of `{total}` are correct. \n Accuracy in our test data is `{(ctr/total)*100}%`')
    test = test.drop(["date"], axis=1, inplace=False)
    test['Predicted'] = predicted_winner
    test


def venueplot(dicVal):
    team1 = dicVal['CSK']
    team2 = dicVal['RCB']
    mtemp = matches[((matches['team1'] == team1) | (matches['team2'] == team1)) & (
        (matches['team1'] == team2) | (matches['team2'] == team2))]
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation='vertical')
    sns.countplot(x='venue', hue='winner', data=mtemp, palette='Set2')
    fig = plt.gcf()
    st.pyplot(fig)
    # st.pyplot(sns.countplot(mtemp['venue']))


def rcbplot(df, opt):
    winCount = 0
    didntCount = 0
    for i in range(577):
        if (df["toss_winner"][i] == opt):
            didntCount = didntCount+1
            if (df["winner"][i] == opt):
                winCount = winCount+1
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    slices = [winCount, (didntCount-winCount)]
    labels = ['Toss & won', 'Toss & lost']
    ax.pie(slices, labels=labels, startangle=90,
           autopct='%1.1f%%', colors=['#F6BC04', '#C12A4D'])
    st.pyplot(fig)


def generalisedplot(df):
    df_fil = df[df['toss_winner'] == df['winner']]
    slices = [len(df_fil), (577-len(df_fil))]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('equal')
    labels = ['Toss & won', 'Toss & lost']
    ax.pie(slices, labels=labels, startangle=90,
           autopct='%1.1f%%', colors=['#F6BC04', '#C12A4D'])
    st.pyplot(fig)


def misc(model, df, dicVal):
    st.header('Other Factors than Team Stats')
    st.subheader(
        'Venue seems to be one of the **important** factors in determining winners followed by toss winning and city')
    st.write("Feature importances: ")
    imp_input = pd.Series(model.feature_importances_,
                          index=predictor_var).sort_values(ascending=False)
    st.write(imp_input)
    option = st.selectbox('Select Team :', ('MI', 'KKR', 'RCB', 'DC', 'CSK', 'RR',
                                            'DD', 'GL', 'KXIP', 'SRH', 'RPS', 'KTK', 'PW'))
    opt = dicVal[option]
    st.subheader(f'Toss Win to match win for `{option}`')
    rcbplot(df, opt)
    st.subheader('Generalised:')
    generalisedplot(df)
    st.write('`Toss` winning `does not gaurantee` a match win from analysis of current stats and thus prediction feature gives less weightage to that')
    st.write('Top 2 team analysis based on number of matches won against each other and how `venue` affects them?')
    st.write('Previously we noticed that `CSK won 79`, `RCB won 70` matches now let us `compare venue` against a match between CSK and RCB')
    st.write('We find that `CSK` has won most matches against RCB in `MA Chidambaram Stadium, Chepauk, Chennai`')
    st.write("`RCB` has not won any match with CSK in stadiums `St George's Park and Wankhede Stadium`, but `won matches with CSK in Kingsmead, New Wanderers Stadium`.")
    venueplot(dicVal)
    st.write('It does prove that chances of CSK winning is more in Chepauk stadium when played against RCB. ***Proves venue is important feature in predictability.***')


def thankyou():
    st.write(''' --- ''')
    st.header('Thank You!🍉')


def main():

    st.title('Cricket Match Prediction 🏏')
    info = f"""A prediction engine that takes in the statistcs of various players from various teams and shows the team that has the best chances of winning.

---
"""
    st.write(info)
    introduction()
    load_data()
    clean_encode()
    data, dicVal = pda()
    model = building_model(data)
    testing(model, dicVal)
    misc(model, data, dicVal)
    thankyou()


if __name__ == '__main__':
    main()
