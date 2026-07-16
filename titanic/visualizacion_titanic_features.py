import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.future.infer_string = False

os.makedirs("graficas", exist_ok=True)

df = pd.read_csv("titanic_cleaned.csv")

print("Info del dataset: ", df.shape)

print("\n Columnas disponibles del nuevo dataset: ")
print(list(df.columns))

print("\n Tipos de datos:  ")
print(df.dtypes)

# Cambio de 'Pclass' a tipo de dato 'category'
df['Pclass'] = df['Pclass'].astype('category')

# Creacion de lista de bins de 'AgeGroup' para cambiar tipo de datos de 'AgeGroup' a tipo de dato 'category' y ordenarlas
orden_edad = ['Child', 'Teenager', 'Adult', 'Middle-aged', 'Senior']
df['AgeGroup'] = pd.Categorical(
    df['AgeGroup'], categories=orden_edad, ordered=True)


print("\n Tipos de datos despues del cambio:  ", df["Pclass"].dtypes)
print(df["Pclass"].value_counts().sort_index())
print("\n Tipos de datos despues del cambio:  ", df['AgeGroup'].dtypes)
print(df['AgeGroup'].value_counts().sort_index())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Survived', hue='Survived',
              palette='Set2', legend=False)
plt.title('Distribución de sobrevivientes')
plt.xlabel('Survived 0 = No sobrevivio, 1 = Sobrevivio')
plt.ylabel('Cantidad de pasajeros')
plt.xticks([0, 1], ['No (0)', 'Sí (1)'])
plt.tight_layout()
plt.savefig('graficas/01_survived.png')
plt.show()

print(df['Survived'].value_counts())
print()
print((df['Survived'].value_counts() / len(df) * 100).round(2))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Parte izquierda

sns.barplot(data=df, x='Sex', y='Survived', hue='Sex',
            palette='Set2', legend=False, ax=axes[0])
axes[0].set_title('Tasa de supervivientes por Genero')
axes[0].set_xlabel('Genero')
axes[0].set_ylabel('Tasa de supervivencia')
axes[0].set_ylim(0, 1)

# Parte derecha
sns.barplot(data=df, x='Pclass', y='Survived', hue='Pclass',
            palette='Set2', legend=False, ax=axes[1])
axes[1].set_title('Tasa de supervivientes por Clase')
axes[1].set_xlabel('Clase')
axes[1].set_ylabel('Tasa de supervivencia')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('graficas/02_sex_pclass.png')
plt.show()

print("Supervivientes por sexo: ")
print(df.groupby('Sex', observed=True)[
      'Survived'].agg(['mean', 'count']).round(2))
print("Supervivientes por clase: ")
print()
print(df.groupby('Pclass', observed=True)[
      'Survived'].agg(['mean', 'count']).round(2))


orden_title = df.groupby('Title')['Survived'].mean(
).sort_values(ascending=False).index


plt.figure(figsize=(9, 5))
sns.barplot(data=df, x='Title', y='Survived', hue='Title',
            order=orden_title, palette='Set2', legend=False)
plt.title('Tasa de supervivientes por Titulo')
plt.xlabel('Titulo')
plt.ylabel('Tasa de supervivencia (promedio)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('graficas/03_title.png')
plt.show()

print("Supervivientes por titulo: ")
print(df.groupby('Title')['Survived'].agg(['mean', 'count']).round(2))


plt.figure(figsize=(9, 5))
sns.barplot(data=df, x='AgeGroup', y='Survived',
            hue='AgeGroup',  palette='Set2', legend=False)
plt.title('Tasa de supervivientes por Grupo de Edad')
plt.xlabel('Grupo de Edad')
plt.ylabel('Tasa de supervivencia(promedio)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('graficas/04_agegroup.png')
plt.show()

print("Supervivientes por grupo de edad: (en orden logico de edad)")
print(df.groupby('AgeGroup', observed=True)[
      'Survived'].agg(['mean', 'count']).round(2))

# Distribucion de  age e isAlone histograma

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Parte izquierda
sns.histplot(data=df, x='Age', bins=30, kde=True,
             ax=axes[0], color='steelblue')
axes[0].set_title('Distribución de edad')
axes[0].set_xlabel('Edad')
axes[0].set_ylabel('Cantidad de pasajeros')

# Parte derecha
sns.barplot(data=df, x='IsAlone', y='Survived', hue='IsAlone',
            palette='Set2', legend=False, ax=axes[1])
axes[1].set_title('Tasa de supervivencia: Viajaba solo?')
axes[1].set_xlabel('IsAlone 0 = Acompañado, 1 = Solo')
axes[1].set_ylabel('Tasa de supervivencia (promedio)')
axes[1].set_ylim(0, 1)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Acompañado (0)', 'Solo (1)'])

plt.tight_layout()
plt.savefig('graficas/05_age_isalone.png')
plt.show()

print("Estadistica de age:")
print(df['Age'].describe().round(2))
print("Supervivientes por si viajaba solo o acompañado: ")
print(df.groupby('IsAlone')['Survived'].agg(['mean', 'count']).round(2))


# Matriz de correlacion - title - sex - embarked
columnas_numericas = df.select_dtypes(include=['int64', 'float64'])

print("Columnas de la matriz de correlacion:")
print(list(columnas_numericas))

correlacion = columnas_numericas.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlacion,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
)

plt.title('Matriz de correlación - Variables numéricas')
plt.tight_layout()
plt.savefig('graficas/06_correlacion.png')
plt.show()

print("\n Correlacion de cada variable con 'Survived (de mayor a menor): ")
print(correlacion['Survived'].sort_values(ascending=False).round(3))