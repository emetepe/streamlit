#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import cufflinks as cf
from sklearn.ensemble import RandomForestClassifier

#app heading
st.write("""
# Análisis de la calidad del vino tinto.
This app predicts the ***Wine Quality*** type!
""")

#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#reading csv file
data=pd.read_csv("winequality-red.csv")
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(data['quality'])

#random forest model
rfc= RandomForestClassifier()
rfc.fit(X, Y)
st.subheader('Wine quality labels and their corresponding index number')
st.write(pd.DataFrame({
   'wine quality': [3, 4, 5, 6, 7, 8 ]}))

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

####

import cufflinks as cf
import warnings
warnings.filterwarnings("ignore")

####### Load Dataset #####################

# wine = load_wine()

# wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

# wine_df["WineType"] = [wine.target_names[t] for t in wine.target ]

# st.set_page_config(layout="wide")

# st.markdown("## Wine Dataset Analysis")   ## Main Title

################# Scatter Chart Logic #################

st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Ingredients :")

ingredients = data.drop(labels=["WineType"], axis=1).columns.tolist()

x_axis = st.sidebar.selectbox("X-Axis", ingredients)
y_axis = st.sidebar.selectbox("Y-Axis", ingredients, index=1)

if x_axis and y_axis:
    scatter_fig = data.iplot(kind="scatter", x=x_axis, y=y_axis,
                    mode="markers",
                    categories="WineType",
                    asFigure=True, opacity=1.0,
                    xTitle=x_axis.replace("_"," ").capitalize(), yTitle=y_axis.replace("_"," ").capitalize(),
                    title="{} vs {}".format(x_axis.replace("_"," ").capitalize(), y_axis.replace("_"," ").capitalize()),
                    )




########## Bar Chart Logic ##################

st.sidebar.markdown("### Bar Chart: Average Ingredients Per Wine Type : ")

avg_wine_df = data.groupby(by=["WineType"]).mean()

bar_axis = st.sidebar.multiselect(label="Bar Chart Ingredient", options=avg_wine_df.columns.tolist(), default=["alcohol","malic_acid"])

if bar_axis:
    bar_fig = avg_wine_df[bar_axis].iplot(kind="bar",
                        barmode="stack",
                        xTitle="Wine Type",
                        title="Distribution of Average Ingredients Per Wine Type",
                        asFigure=True,
                        opacity=1.0,
                        );
else:
    bar_fig = avg_wine_df[["alcohol"]].iplot(kind="bar",
                        barmode="stack",
                        xTitle="Wine Type",
                        title="Distribution of Average Alcohol Per Wine Type",
                        asFigure=True,
                        opacity=1.0,
                        );

################# Histogram Logic ########################

st.sidebar.markdown("### Histogram: Explore Distribution of Ingredients : ")

hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=ingredients, default=["malic_acid"])
bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=1)

if hist_axis:
    hist_fig = data.iplot(kind="hist",
                             keys=hist_axis,
                             xTitle="Ingredients",
                             bins=bins,
                             title="Distribution of Ingredients",
                             asFigure=True,
                             opacity=1.0
                            );
else:
    hist_fig = data.iplot(kind="hist",
                             keys=["alcohol"],
                             xTitle="Alcohol",
                             bins=bins,
                             title="Distribution of Alcohol",
                             asFigure=True,
                             opacity=1.0
                            );


#################### Pie Chart Logic ##################################

wine_cnt = data.groupby(by=["WineType"]).count()[['alcohol']].rename(columns={"alcohol":"Count"}).reset_index()

pie_fig = wine_cnt.iplot(kind="pie", labels="WineType", values="Count",
                         title="Wine Samples Distribution Per WineType",
                         hole=0.4,
                         asFigure=True)


##################### Layout Application ##################

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        scatter_fig
    with col2:
        bar_fig


container2 = st.container()
col3, col4 = st.columns(2)

with container2:
    with col3:
        hist_fig
    with col4:
        pie_fig
