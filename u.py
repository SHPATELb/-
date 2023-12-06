import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


st.write(
    """
# Предсказание с помощью моделей классификаторов
"""
)

st.sidebar.header("Параметры, вводимые пользователем")


@st.cache
def load_data(file):
    df = pd.read_excel(file)
    return df


uploaded_file = st.sidebar.file_uploader("Загрузите файл типа .xlsx", type="xlsx")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    Z = df.drop(["wavenumber"], axis=1).values
    O = np.zeros(Z.shape[0])
    classifiers = {
        "Случайный лес": RandomForestClassifier(),
        "Метод ближайших соседей": KNeighborsClassifier(),
        "Наивный Байесовский классификатор": GaussianNB(),
        "Метод опорных векторов": LDA(),
        "Проекция на латентные структуры": PLSRegression(),
    }
    classifier_name = st.sidebar.selectbox(
        "Выберите классификатор", list(classifiers.keys())
    )

    clf = classifiers[classifier_name]
    clf.fit(Z, O)
    scores = cross_val_score(clf, Z, O, cv=5)

    model = clf.fit(Z, O)
    Y_pred = model.predict(Z)

    number_of_true_answers = 0
    for Y_t, Y_p in zip(O, Y_pred):
        if Y_t == round(Y_p):
            number_of_true_answers += 1

    n = f"{number_of_true_answers / len(O) * 100}%"

    # st.subheader("Предсказание")
    # st.write(iris.target_names[prediction])

    st.subheader("Вероятность предсказания")
    st.write(n)

    st.subheader("Устойчивость модели")
    st.write(scores)

else:
    sepal_length = st.sidebar.slider("длина чашелистика", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("ширина чашелистика", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider("длина лепестка", 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider("ширина лепестка", 0.1, 2.5, 0.2)
    data = {
        "длина чашелистика": sepal_length,
        "ширина чашелистика": sepal_width,
        "длина лепестка": petal_length,
        "ширина лепестка": petal_width,
    }
    df = pd.DataFrame(data, index=[0])

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    clf = RandomForestClassifier()
    clf.fit(x, y)

    prediction = clf.predict(df.values)
    prediction_proba = clf.predict_proba(df.values)
    st.subheader("Метки классов и соответствующий им порядковый номер")
    st.write(iris.target_names)

    st.subheader("Предсказание")
    st.write(iris.target_names[prediction])

    st.subheader("Вероятность предсказания")
    st.write(prediction_proba)


st.subheader("Параметры, вводимые пользователем")
st.write(df)
