import pandas as pd

pd.options.future.infer_string = False
df = pd.read_csv("clientes gimnasio.csv")

# Mostrar forma, encabezado y tipos de datos
print("Forma del DataFrame:", df.shape)
print("Encabezado del DataFrame:")
print(df.head())
print("Tipos de datos del DataFrame:")
print(df.dtypes)

# Mostrar porcentaje de nulos

print(df.isnull().sum())
print("\nPorcentaje de valores nulos:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Trabajar con valores de edad

print("Agregamos mediana de edad para los valores nulos de la columna 'Edad' ya que es inmune a outliers")
df['Edad'] = df['Edad'].fillna(df['Edad'].median())

# Trabajar con nulos de sucursal
print("Agregamos categoria 'Unknown' para los valores nulos de la columna 'Sucursal'")
df['Sucursal'] = df['Sucursal'].fillna('Unknown')

print("Al ser un 3.17% no afectaria tanto el analisis, pero opte por reemplazarlo por una categoria"
      "'Unknown' para no perder informacion de los clientes que no tienen sucursal asignada"
      "o para evitar inflar alguna sucural en particular")

# Trabajar con nulos de email
print("Agregamos 'Not registered' para los valores nulos de la columna 'Email'")
df['Email'] = df['Email'].fillna('Not registered')

print("En este caso para un analisis comun se podrian borrar ya que no es concluyente lo que se puede analizar con email"
      "pero para un analisis de marketing es importante tener la mayor cantidad de clientes posibles, por lo que se opto por reemplazarlo por 'Not registered'"
      "para facilitar interpretacion de los datos y no perder informacion de clientes que no tienen email registrado"
      "Adicionalmente, se recomendaria una campaña para actualizar datos ya que los clientes en un gimnasio tiene una gran recurrencia")

print("\nPost Data: Evitar tomar la decision por tu cuenta, siempre presentar opciones al cliente, punto de vista de acuerdo a la informacion que se tiene"
      " y ponerse en los pies del cliente para evitar afectar la relacion con el mismo, ya que es un negocio de recurrencia y no de una sola venta")

# Mostrar nulos restantes
print("Nulos restantes: ", df.isnull().sum().sum())
print(df.shape)

# Eliminar duplicados
print("Filas antes de eliminar duplicados: ", len(df))
df = df.drop_duplicates()
print("Filas después de eliminar duplicados: ", len(df))

# Corregir tipos de datos
print("Corregimos tipos de datos de la columna 'Plan' de int64 a category")
df['Plan'] = df['Plan'].astype('category')
print(df.dtypes)
