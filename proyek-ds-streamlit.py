import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Judul Aplikasi
st.title('Prediksi Cuaca')

df = pd.read_csv('fix.csv')

# Input Kelembaban
in_reg = st.slider('Masukkan Nilai Kelembaban (%)', min_value=0, max_value=100, value=0)

# Prediksi Suhu
y = df['rata-rata_suhu'].values
x = df['rata-rata_kelembaban_persen'].values.reshape(-1,1)
lm = LinearRegression()
lm.fit(x, y)
out_reg = lm.predict(np.array([in_reg]).reshape(-1,1))[0]

# Output Prediksi Suhu
st.write(f'Nilai Kelembaban: {in_reg} %')
st.write(f'Nilai Prediksi Suhu: {round(out_reg, 1)}°C')

# Prediksi Cuaca
X = df[['rata-rata_suhu', 'rata-rata_kelembaban_persen']]
Y = df['cuaca']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Memberikan nama kolom yang valid untuk objek X
X_train.columns = ['suhu', 'kelembaban']
X_test.columns = ['suhu', 'kelembaban']

# Membuat objek Decision Tree Classifier
model = DecisionTreeClassifier()

# Melatih model dengan data latih
model.fit(X_train, y_train)

# Memprediksi data uji
y_pred = model.predict(X_test)

def predict(tree, X_new):
    # Jika node saat ini merupakan node daun, return label kelas pada node
    if tree.left is None and tree.right is None:
        return tree.label
    # Jika fitur pada node memenuhi kriteria pemisahan data baru, rekursif panggil fungsi predict
    # dengan node anak kiri dan data baru yang telah dipisahkan
    if X_new[tree.feature] <= tree.threshold:
        return predict(tree.left, X_new)
    # Jika fitur pada node tidak memenuhi kriteria pemisahan data baru, rekursif panggil fungsi predict
    # dengan node anak kanan dan data baru yang telah dipisahkan
    else:
        return predict(tree.right, X_new)

# Membangun model Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Melakukan prediksi pada data baru
X_new = [[out_reg, in_reg]]  # Contoh data baru dengan suhu 65 dan kelembaban 80
prediction = tree.predict(X_new)


# Output Prediksi Cuaca
st.write(f'Diprediksikan dengan Kondisi Cuaca: {prediction[0]}')
