import pandas as pd

# Mostrar el contenido del archivo CSV
df = pd.read_csv('titanic.csv')

print("Forma del dataset:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print("\nTipos de datos:")
print(df.dtypes)

# Diagnosticar los valores nulos en el dataset

# Crear variables para ver nulos en intervalos absoluto y en porcentaje
nulos = df.isnull().sum()
porcentaje_nulos = (nulos / len(df) * 100).round(2)

print("\nValores nulos por columna:")
print(nulos)
print("\nPorcentaje de valores nulos por columna:")
print(porcentaje_nulos)

# Aplicar mediana Age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Aplicar moda Embarked
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Borrar la columna Cabin
df = df.drop(columns=['Cabin'])

# Mostrar si hay valores nulos después de la limpieza
print("Cantidad de valores nulos después de la limpieza:", df.isnull().sum().sum())

print("Filas antes de eliminar duplicados:", len(df))
df = df.drop_duplicates()
print("Cantidad de filas después de eliminar duplicados:", len(df))

# cambbiar tipo de dato
df['Pclass'] = df['Pclass'].astype('category')


print(df.isnull().sum())


# Feature engineering

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(df[['SibSp', 'Parch', 'FamilySize']].head())


# Variables binarias

df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print(df['IsAlone'].value_counts())

# Extraer el título del nombre

df['Title'] = df['Name'].str.extract(r',\s*([A-Za-z]+)\.?\s')

print(df['Title'].value_counts())

# Unificar categorias raras

df['Title'] = df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs',
    'Dr': 'Rare',
    'Rev': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Capt': 'Rare',
    'Sir': 'Rare',
    'Lady': 'Rare',
    'the': 'Rare',
    'Jonkheer': 'Rare',
    'Don': 'Rare'
})

print(df['Title'].value_counts())

# Discretizar la edad en grupos

df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior']
)

print(df[['Age', 'AgeGroup']].head(10))
print(df['AgeGroup'].value_counts())

print("DATASET FINAL LIMPIO Y CON FEATURES ENGINEERING:")
print(f"\nForma Final: {df.shape[0]} filas y {df.shape[1]} columnas")
print(f"\n Columnas actuales: \n{list(df.columns)}")
print(f"\n Valores nulos restantes: \n{df.isnull().sum().sum()}")
print(f"\n Vista previa del dataset final: \n{df.head()}")

# Guardar el resultado en un nuevo archivo

df.to_csv('titanic_cleaned.csv', index=False)
