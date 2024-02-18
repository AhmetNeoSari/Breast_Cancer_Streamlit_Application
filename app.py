import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# keeps the plots in one place. calls image as static pngs
# matplotlib inline 
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.gridspec as gridspec # subplots
# import mpld3 as mpl

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
"""
sayfanın gerekli yerlerine başlıklar ekle !!!!!!!!!!!!!!!!!!!!!!!!!!!1 subheader header
"""

# try:
#     df = pd.read_csv("archive/data.csv",header = 0)
# except:
#     print("File not found")

class App:
    def __init__(self):
        self.dataset_name = None

        self.Init_Streamlit_Page()

        self.df = None
        self.column_names = None
        self.params = dict()
        self.clf = None
        self.X, self.Y = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None  

    def run(self):
        self.get_dataset()
        self.preprocess_data()
        self.plot_data()
        # self.add_parameter_ui()
        # self.generate()

    def Init_Streamlit_Page(self):
        st.title('Ahmet Sarı YZUP - Streamlit App')
        
        # st.subheader('Select Dataset and Classifier')

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            (['something else','Breast Cancer'])
        )

        st.write(f"## {self.dataset_name} Dataset")
        if self.dataset_name == "something else":
            st.write("Please select the correct dataset from the dropdown menu")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naive Bayes')
        )    

    def get_dataset(self):
        if self.dataset_name == "Breast Cancer":
            try:
                self.df = pd.read_csv("data.csv", header = 0)

                st.dataframe(self.df.head(10))
                st.write(f"Shape: {self.df.shape}")
                st.write(f"Columns Names: {self.df.columns.values}")


            except:
                st.write("ERROR: File not found")
                raise FileNotFoundError

    def get_classifier(self):
        self.classifier_name = st.selectbox("Select Classifier", ["Logistic Regression", "Random Forest", "Decision Tree"])

    def get_params(self):
        # get parameters using gridSearch
        pass

    def preprocess_data(self):
        if self.dataset_name == "Breast Cancer":
            if 'id' in self.df.columns:
                self.df.drop("id", axis=1, inplace=True)     

            self.df.dropna(axis=1, how='all', inplace=True)

            st.write("After removing id and Unnamed: 32 columns from the dataset, the last 10 rows of the dataset are:")
            st.write(self.df.tail(10))
            st.write(f"Shape: {self.df.shape}")

            for column in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[column].dtype):
                    mean_value = self.df[column].mean()
                    self.df[column].fillna(mean_value, inplace=True)

            self.df['diagnosis'] = self.df['diagnosis'].map({'M':1, 'B':0})
            self.X = self.df.drop("diagnosis", axis = 1)
            self.Y = self.df['diagnosis']

            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = 0.3, random_state = 101)

            # st.write("self.x", self.X.shape)
            # st.write(self.X)
            # st.write("self.y", self.Y.shape)
            # st.write(self.Y)

            st.write(f"X_train: {self.X_train.shape}, X_test: {self.X_test.shape}, Y_train: {self.Y_train.shape}, Y_test: {self.Y_test.shape}")
            st.write(f"Columns Names: {self.df.columns.values}")

        else:
            st.write("Something else")

    def generate(self):
        pass

    def plot_data(self):
        if self.dataset_name == "Breast Cancer":
            # Veriyi hazırla (sadece 'diagnosis', 'radius_mean', 'texture_mean' sütunlarını kullanacağız)
            selected_columns = ['diagnosis', 'radius_mean', 'texture_mean']
            selected_data = self.df[selected_columns]

            st.title('Korelasyon Matrisi ve Scatter Plot')

            # Korelasyon matrisini çizdir
            correlation_matrix = selected_data.corr()
            st.subheader('Korelasyon Matrisi')
            st.write(correlation_matrix)

            # Heatmap çizdir
            st.pyplot(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f").figure)

            # Malignant ve Benign örnekleri ayır
            malignant_data = selected_data[selected_data['diagnosis'] == 1]
            benign_data = selected_data[selected_data['diagnosis'] == 0]

            # Scatter plot çizdir
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='radius_mean', y='texture_mean', data=benign_data, color='green', label='İyi')
            sns.scatterplot(x='radius_mean', y='texture_mean', data=malignant_data, color='red', label='Kötü')
            plt.title('Radius Mean vs Texture Mean')
            plt.xlabel('Radius Mean')
            plt.ylabel('Texture Mean')
            plt.legend()

            # Scatter plot'u göster
            st.subheader('Scatter Plot')
            st.pyplot(plt)







