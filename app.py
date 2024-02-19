import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt # side-stepping mpl backend
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from numpy import arange

""" 
this project is a simple streamlit app that uses the breast cancer dataset to train and evaluate a classifier

"""


class App:
    def __init__(self):
        self.dataset_name = None
        self.df = None
        self.column_names = None
        self.params = dict()
        self.clf = None
        self.X, self.Y = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None  

        self.Init_Streamlit_Page()
   
    def run(self):
        self.get_dataset()
        self.preprocess_data()
        self.plot_data()
        self.generate()

    def Init_Streamlit_Page(self):
        st.title('Ahmet Sarı YZUP - Streamlit App')
    
        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            (['Unknown Dataset','Breast Cancer'])
        )

        if self.dataset_name == "Unknown Dataset":
            st.warning("Please select the correct dataset from the dropdown menu")
        else:
            st.write(f"## {self.dataset_name} Dataset")
            

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
            
    def _get_classifier(self):
    
        if self.classifier_name == 'KNN':
            try:
                self.clf = KNeighborsClassifier()
                param_grid = {'n_neighbors': arange(1, 20, 1)}

                grid_search = GridSearchCV(self.clf, param_grid, cv=10, scoring="accuracy")
                grid_search = grid_search.fit(self.X_train, self.Y_train)
                # st.write(f"Best Parameters for KNeighborsClassifier: {grid_search.best_params_}")

                self.clf = grid_search.best_estimator_  # En iyi modeli seç

            except Exception as e:
                st.write(f"Error: {e}")

        elif self.classifier_name == 'SVM':
            self.clf = SVC(probability=True)  # probability=True, predict_proba kullanmak için
            param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
            grid_search = GridSearchCV(self.clf, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.Y_train)

            # st.write(f"Best Parameters for SVM: {grid_search.best_params_}")

            # En iyi parametrelerle modeli tekrar eğit
            self.clf = grid_search.best_estimator_
    
        else:
            self.clf = GaussianNB()


    def generate(self):
        if self.X_test is not None and self.Y_test is not None and self.X_train is not None and self.Y_train is not None:
            try:
                self._get_classifier()
                self.clf.fit(self.X_train, self.Y_train)
                self._predict_and_evaluate()
            except Exception as e:
                st.write(f"Error: {e}")
     
    def _predict_and_evaluate(self):
        try:
            predictions = self.clf.predict(self.X_test)

            st.title('Model Evaluation Metrics')
            
            # Accuracy
            accuracy = metrics.accuracy_score(self.Y_test, predictions)
            st.write(f'Accuracy: {accuracy:.4f}')

            # Precision
            precision = metrics.precision_score(self.Y_test, predictions)
            st.write(f'Precision: {precision:.4f}')

            # Recall
            recall = metrics.recall_score(self.Y_test, predictions)
            st.write(f'Recall: {recall:.4f}')

            # F1-Score
            f1 = metrics.f1_score(self.Y_test, predictions)
            st.write(f'F1-Score: {f1:.4f}')

            # Confusion Matrix
            conf_matrix = confusion_matrix(self.Y_test, predictions)
            self._plot_conf_matrix(conf_matrix)

        except:
            st.warning('Please select a dataset and classifier, and make sure the model is trained.')



    def preprocess_data(self):
        if self.dataset_name == "Breast Cancer":
            if 'id' in self.df.columns:
                self.df.drop("id", axis=1, inplace=True)     

            self.df.dropna(axis=1, how='all', inplace=True)

            for column in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[column].dtype):
                    mean_value = self.df[column].mean()
                    self.df[column].fillna(mean_value, inplace=True)

  
            self.df['diagnosis'] = self.df['diagnosis'].map({'M':1, 'B':0})
            self.X = self.df.drop("diagnosis", axis = 1)
            self.Y = self.df['diagnosis']

            st.subheader("After Preprocessing:")
            st.write(self.df.tail(10))
            st.write(f"Shape of dataset: {self.df.shape}")


            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = 0.2, random_state = 101)

            st.write(f"X_train: {self.X_train.shape}, X_test: {self.X_test.shape}, Y_train: {self.Y_train.shape}, Y_test: {self.Y_test.shape}")

    def _plot_conf_matrix(self, conf_matrix):

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
                    yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'])
        plt.xlabel('y_pred')
        plt.ylabel('y_true')
        st.subheader('Confusion Matrix')
        st.pyplot(plt)



    def plot_data(self):
        if self.dataset_name == "Breast Cancer":
            # Veriyi hazırla (sadece 'diagnosis', 'radius_mean', 'texture_mean' sütunlarını kullanacağız)
            selected_columns = ['diagnosis', 'radius_mean', 'texture_mean']
            selected_data = self.df[selected_columns]

            st.subheader('Correlation Matrix and Scatter Plot')

            # Korelasyon matrisini çizdir
            correlation_matrix = selected_data.corr()
            st.subheader('Correlation Matrix')
            # st.write(correlation_matrix)  

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







