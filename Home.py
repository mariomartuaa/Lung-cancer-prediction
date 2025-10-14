import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib as joblib

st.set_page_config(
    page_title="Home",
    page_icon="🫁",
    layout="wide"
)
        
st.title('Lung Cancer Dataset')
st.markdown('- Library: numpy, pandas, streamlit, matplotlib, seaborn')
st.markdown('- Dataset: https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link')

df = pd.read_csv('cancer patient data sets.csv')

st.write('Lung cancer is the leading cause of cancer death worldwide, accounting for 1.59 million deaths in 2018. The majority of lung cancer cases are attributed to smoking, but exposure to air pollution is also a risk factor. A new study has found that air pollution may be linked to an increased risk of lung cancer, even in nonsmokers.\n\nThe study, which was published in the journal Nature Medicine, looked at data from over 462,000 people in China who were followed for an average of six years. The participants were divided into two groups: those who lived in areas with high levels of air pollution and those who lived in areas with low levels of air pollution.\n\nThe researchers found that the people in the high-pollution group were more likely to develop lung cancer than those in the low-pollution group. They also found that the risk was higher in nonsmokers than smokers, and that the risk increased with age.\n\nWhile this study does not prove that air pollution causes lung cancer, it does suggest that there may be a link between the two. More research is needed to confirm these findings and to determine what effect different types and levels of air pollution may have on lung cancer risk')

st.dataframe(data=df[:10])

st.subheader('Total of Lung Cancer Patients Based on Age')
fig, ax = plt.subplots(figsize=(25,7))
sns.countplot(x="Age", data=df)
st.pyplot(fig)

st.subheader('Total of Lung Cancer Patients Based on Gender')
Male = []
Female = []
label = df['Level'].unique()

for j in label:
    Male.append(df[(df['Level'] == j) & (df['Gender'] == 1)]['Level'].count())
    Female.append(df[(df['Level'] == j) & (df['Gender'] == 2)]['Level'].count())

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,6))
ax[0].pie(x=Male, autopct='%1.1f%%')
ax[0].set_title('Male cancer level')
ax[1].pie(x=Female, autopct='%1.1f%%')
ax[1].set_title('Female cancer level')
ax[0].legend(label,loc="upper left",fontsize=15)
st.pyplot(fig)

st.subheader('Heatmap for all columns')
fig, ax = plt.subplots(figsize=(25, 10))
corr = df.apply(lambda x: pd.factorize(x)[0]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(fig)