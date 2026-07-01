import pandas as pd

pd.options.future.infer_string = False
df = pd.read_csv("clientes gimnasio.csv")

# Mostrar forma, encabezado y tipos de datos
print("Forma del DataFrame:", df.shape)
print("Encabezado del DataFrame:")
print(df.head())
print("Tipos de datos del DataFrame:")
print(df.dtypes)
