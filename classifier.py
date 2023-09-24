import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings

file_path = 'Datos.xlsx'
fondos = pd.read_excel(file_path)

st.title('Modelo de predicciones para fondos de inversión')

x = fondos.drop('Target', axis = 1)
y = fondos['Target']

seed = 49

X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=seed
    )

#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Crear el modelo KNN
k = 3  # Número de vecinos a considerar (ajusta este valor según tus necesidades)
knn_model = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
knn_model.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn_model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

def obtenerTarget( ft1, ft2, ft3):
    custom_data = np.array([[ft1, ft2, ft3]])
    prediccion = knn_model.predict(custom_data)
    return prediccion[0]

warnings.filterwarnings("ignore")

tipoFondo = {
    1: "NTECT",
    2: "NTEDIG",
    3: "NTEPZO1",
    4: "NTED",
    5: "NETDLS",
    6: "NTEDLS+",
    7: "NTEIPC+",
    8: "NTEESG"
}

ft1 = float(st.text_input ( "Disponibilidad de inversión", "876"))
ft2 = float(st.text_input ( "Plazo para conseguir la meta", "1"))
ft3 = float(st.text_input ( "Meta", "1750"))

resultado = tipoFondo[obtenerTarget(ft1,ft2,ft3)]
st.text_area("Tipo de fondo",resultado + "", height =10)
print ( obtenerTarget(ft1,ft2,ft3) )
