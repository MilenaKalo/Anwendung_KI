import pandas as pd
import matplotlib.pyplot as plt
# CSV-Datei einlesen
data = pd.read_csv('/Users/milena/Downloads/aki/C-LSTM-text-classification/data/data.csv')

# Ausgabe der ersten Zeilen
print(data.head())

# Informationen über die Spalten und Datentypen
print(data.info())

# Statistische Kennzahlen
print(data.describe())

# zählen der Werte in der Spalte 'label'
print(data['label'].value_counts())
#Visualisierung der Daten
# Hier ein Beispiel für ein Histogramm der Spalte 'Spalte1'
data['label'].hist()
plt.show()

# Datenbereinigung
# Überprüfung auf fehlende Werte
print(data.isnull().sum())

# Überprüfung auf Duplikate
print(data.duplicated().sum())

