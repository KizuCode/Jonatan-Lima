# Importar librerías para el análisis y visualización de datos
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keyboard

# Definición de la función para cargar el archivo CSV desde una ruta especificada
def cargar_csv(ruta_archivo):
    try:
        archivo_csv = pd.read_csv(ruta_archivo, sep=',')
        print('El archivo se ha cargado correctamente')
        return archivo_csv
    except FileNotFoundError:
        raise FileNotFoundError(f'El archivo no se encuentra en la ubicación especificada: {ruta}')

# Carga del archivo CSV
ruta = 'winemag-data-130k-v2.csv'
vinos_csv = cargar_csv(ruta)

print('\t\t\t\t\t\t\t\t--//--\n')

# Primeras filas del DataFrame 'vinos_csv'
print('Comprobación de primeras filas del archivo CSV:\n')
print(vinos_csv.head())

# Conteo de la cantidad de valores nulos (NaN) y ordenamiento del conteo en forma descendente
contador_nan = vinos_csv.isnull().sum()
contador_nan = contador_nan.sort_values(ascending=False)

# Creación de un nuevo DataFrame y renombre de columnas para mayor claridad
df_contador_nan = contador_nan.reset_index()
df_contador_nan.columns = ['Variable', 'Total de NaN']

print('\t\t\t\t\t\t\t\t--//--\n')

# Muestreo del DataFrame con el conteo de NaN
print('Cantidad de valores NaN en cada variable:\n')
print(df_contador_nan)

print('\t\t\t\t\t\t\t\t--//--\n')

# Filtrado del DataFrame para obtener las filas donde tanto 'country' como 'province' son NaN
datos_faltantes_country_province = vinos_csv[vinos_csv['country'].isnull() & vinos_csv['province'].isnull()]

# Obtención del número de filas con valores NaN en 'country' y 'province'
# Calculo del porcentaje de registros con NaN sobre el total de datos
total_filas = vinos_csv.shape[0]
datos_faltantes_filas = datos_faltantes_country_province.shape[0]
porcentaje_NaN_sobre_total = datos_faltantes_filas / total_filas * 100

# Visualización de los resultados con valores NaN en 'country' y 'province'
print(f"Se encontraron {datos_faltantes_filas} registros en donde el campo 'country' y 'province' tienen valores NaN simultaneamente")
print(f'Estos registros representan el {round(porcentaje_NaN_sobre_total, 2)}% del total de datos')

# Eliminamos la columna 'Unnamed: 0' del DataFrame, que es un índice innecesario y 'region_2' el cual tiene la mayor número de nulos
vinos_csv = vinos_csv.drop(columns=['Unnamed: 0', 'region_2'])

# Eliminación de las filas del DataFrame donde 'country' y 'province' tienen valores NaN
# Se carga en otra variable el resultado de la eliminación
vinos_csv_nuevo = vinos_csv.dropna(subset=['country', 'province'])

print('\t\t\t\t\t\t\t\t--//--\n')

# Información general sobre el DataFrame 'vinos_csv_nuevo'
vinos_csv_nuevo.info()

print('\t\t\t\t\t\t\t\t--//--\n')

# Filtramos las filas duplicadas en el DataFrame 'vinos_csv_nuevo' y las almacenamos en la variable 'filas_repetidas'.
# Se utiliza 'duplicated(keep=False)' para marcar todas las ocurrencias duplicadas, no solo la primera.
filas_repetidas = vinos_csv_nuevo[vinos_csv_nuevo.duplicated(keep=False)]

# Se procede a realizar el calculo de representacion de las filas duplicadas con respecto al total de filas
total_filas = vinos_csv.shape[0]
total_filas_repetidas = filas_repetidas.shape[0]
porcentaje_NaN_sobre_total = total_filas_repetidas / total_filas * 100

# print de los resultados de las filas repetidas
print(f'Se encontraron {total_filas_repetidas} filas repetidas')
print(f'Estas filas representa un {round(porcentaje_NaN_sobre_total, 2)}% del total de filas')

print('\t\t\t\t\t\t\t\t--//--\n')

# Eliminamos las filas duplicadas en el DataFrame 'vinos_csv_nuevo'.
vinos_csv_nuevo = vinos_csv_nuevo.drop_duplicates()

# Mostramos la información del DataFrame 'vinos_csv_nuevo' después de eliminar las filas duplicadas.
vinos_csv_nuevo.info()

print('\t\t\t\t\t\t\t\t--//--\n')

# Calculo el puntaje promedio por país
mean_points_x_country = vinos_csv_nuevo.groupby('country')['points'].mean().sort_values(ascending=False).reset_index()

# Calculo la desviación estándar por país
std_points_x_country = vinos_csv_nuevo.groupby('country')['points'].std().reset_index()

# Union de ambos dataframes en uno solo
df_points_mean_std_x_country = pd.merge(mean_points_x_country, std_points_x_country, on='country', suffixes=('_mean', '_std'))

# Mostrar resultados
print('Países con los mejores puntajes promedio y su desviación estándar:\n')
print(df_points_mean_std_x_country.round(2))

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Puntos medios por país - Gráfico de linea\n')

# Puntos medios por país - Gráficos de lineas
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_points_mean_std_x_country, x='country', y='points_mean', marker='o', color='#8B0000')
plt.xticks(rotation=90)
plt.title('Puntos medios por país')
plt.xlabel('Country')
plt.ylabel('points_mean')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# DataFrame temporal sin nulos en 'price' y 'points' y sin precios cero
df_precio_calidad = vinos_csv_nuevo.dropna(subset=['price', 'points'])
df_precio_calidad = df_precio_calidad[df_precio_calidad['price'] > 0]

# Calculo la relación precio-calidad
df_precio_calidad['precio_calidad'] = df_precio_calidad['points'] / df_precio_calidad['price']

# Agrupacion por país y calcular el promedio del índice precio-calidad
mean_precio_calidad_x_country = df_precio_calidad.groupby('country')['precio_calidad'].mean().sort_values(ascending=False).reset_index()

# Mostrar los resultados
print('Paises y su relación precio-calida:\n')
print(mean_precio_calidad_x_country.round(2))

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Relación precio-calidad por país - Gráfico de linea\n')

# Relación precio-calidad por país - Gráficos de lineas
plt.figure(figsize=(12, 6))
sns.lineplot(data=mean_precio_calidad_x_country, x='country', y='precio_calidad', marker='o', color='#8B0000')
plt.xticks(rotation=90)
plt.title('Relación precio-calidad por país')
plt.xlabel('Country')
plt.ylabel('precio_calidad')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# Conteo del número de reseñas por vino
all_vinos = vinos_csv_nuevo['title'].value_counts().reset_index()
all_vinos.columns = ['title', 'num_reseñas']

# Union el conteo de reseñas en un nuevo DataFrame
df_aux = vinos_csv_nuevo
df_nuevo_all_vinos = pd.merge(df_aux, all_vinos, on='title')

# Ordenar los vinos por el número de reseñas
df_nuevo_all_vinos = df_nuevo_all_vinos.sort_values(by='num_reseñas', ascending=False)

# Analizar la distribución de calidad (puntos) y precio para los vinos con más reseñas
df_price_points_all_vinos = df_nuevo_all_vinos.groupby('title').agg({
    'points': ['mean', 'std'],
    'price': ['mean', 'std'],
    'num_reseñas': 'max'
}).reset_index()

# Renombrar las columnas para mayor claridad
df_price_points_all_vinos.columns = ['title', 'mean_points', 'std_points', 'mean_price', 'std_price', 'num_reseñas']

# Mostrar los resultados
df_price_points_all_vinos = df_price_points_all_vinos.sort_values(by='num_reseñas', ascending=False).drop_duplicates(subset='title')

print('Resultados de los vinos con sus puntajes promedio obtenidos y la cantidad de reseñas realizadas:\n')
print(df_price_points_all_vinos.round(2))

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Relación entre Precio Medio y Puntos Medios de los Vinos - Gráfico de linea\n')

# Relación entre Precio Medio y Puntos Medios de los Vinos - Gráficos de lineas
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_price_points_all_vinos, x='mean_price', y='mean_points', marker='o', color='#8B0000')
plt.grid(linestyle='--', alpha=0.7)
plt.xlabel('mean_price')
plt.ylabel('mean_points')
plt.title('Relación entre Precio Medio y Puntos Medios de los Vinos')
plt.tight_layout()
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# Eliminar filas con valores nulos en 'price'
df_price_x_country = vinos_csv_nuevo.dropna(subset='price')

# Agrupar por país y calcular estadísticas descriptivas del precio
country_x_price = df_price_x_country.groupby('country')['price'].describe()

# Mostrar los resultados
print('País y su variación de precios:\n')
print(country_x_price.round(2))

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Variación de precios de vinos por país - Gráfico BoxPlot\n')

# Variación de precios de vinos por país - Gráfico BoxPlot
plt.figure(figsize=(15, 8))
sns.boxplot(data=df_price_x_country, x='country', y='price', hue='country', palette='flare')
plt.title('Variación de precios de vinos por país')
plt.yscale('log')
plt.xlabel('country')
plt.ylabel('price')
plt.grid(linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# Cálculo del puntaje global de cada varietal
variety_mean_x_points = vinos_csv_nuevo.groupby('variety')['points'].mean().sort_values(ascending=False).reset_index()
variety_mean_x_points.columns = ['variety', 'mean_points']

# Mostrar los resultados
print('Promedio de puntos por varietal:\n')
print(variety_mean_x_points.round(2))

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico 1 - Boxplot 1 sobre distribución de "points" por la variable "variety" con el mayor puntaje de la media de "points" - Top 10 - Gráfico BoxPlot\n')

# Selección del Top 10 paises
top_10_variety = variety_mean_x_points.head(10)
top_10_variety = pd.DataFrame(top_10_variety)

# Boxplot sobre distribución de "points" por la variable "variety" con el mayor puntaje de la media de "points" - Top 10 - Gráfico BoxPlot
plt.figure(figsize=(10,8))
sns.boxplot(x='points', y='variety', data=vinos_csv_nuevo[vinos_csv_nuevo['variety'].isin(top_10_variety['variety'])], hue='variety', palette='coolwarm', legend=False)
plt.grid(linestyle='--', alpha=0.7)
plt.title('Boxplot 1 sobre distribución de "points" por la variable "variety" con el mayor puntaje de la media de "points" - Top 10')

plt.tight_layout()
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico 2 - Boxplot 2 sobre distribución de "points" por la variable "variety" mas utilizadas - Top 10 - Gráfico BoxPlot\n')

# Selección del Top 10 paises
top_10_variety = vinos_csv_nuevo.groupby('variety').size().sort_values(ascending=False).index[:10]
top_10_variety = pd.DataFrame(top_10_variety)

plt.figure(figsize=(10,8))
sns.boxplot(x='points', y='variety', data=vinos_csv_nuevo[vinos_csv_nuevo['variety'].isin(top_10_variety['variety'])], hue='variety', palette='coolwarm', legend=False)
plt.grid(linestyle='--', alpha=0.7)
plt.title('Boxplot 2 sobre distribución de "points" por la variable "variety" mas utilizadas - Top 10')

plt.tight_layout()
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# Identificar el variety más utilizado por país
variety_mas_usada_x_country = vinos_csv_nuevo.groupby('country')['variety'].agg(lambda x: x.value_counts().index[0])

# Calcular el puntaje promedio del variety más utilizado por país
mean_points_variety = vinos_csv_nuevo[vinos_csv_nuevo['variety'].isin(variety_mas_usada_x_country)].groupby('country')['points'].mean()

# Calcular la variabilidad del puntaje del variety más utilizado por país
std_points_variety = vinos_csv_nuevo[vinos_csv_nuevo['variety'].isin(variety_mas_usada_x_country)].groupby('country')['points'].std()

# Combinar los resultados en un dataframe
df_country_variety_mean_std = pd.DataFrame({
    'variety': variety_mas_usada_x_country,
    'mean_points': mean_points_variety,
    'std_points': std_points_variety
})

# Mostrar los resultados
print('Varietal más usado por país y su puntaje:\n')
print(df_country_variety_mean_std.sort_values(by='mean_points', ascending=False).round(2))

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Puntaje Promedio y desviación estándar del varietal más utilizado por país - Gráfico de lineas\n')

# Creamos nuevo DataFrame 'resultados' para no alterear el anterior utilizado para el calculo
resultados = df_country_variety_mean_std

# Crear un nuevo dataframe para agrupar los datos
resultados_melted = pd.melt(resultados.reset_index(), id_vars=['country'], value_vars=['mean_points', 'std_points'], 
                            var_name='tipo', value_name='valor')

plt.figure(figsize=(12, 6))
sns.lineplot(data=resultados_melted, x='country', y='valor', hue='tipo')
plt.xlabel('País')
plt.ylabel('Valor')
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=90)
plt.title('Puntaje Promedio y desviación estándar del varietal más utilizado por país')
plt.legend(title='Tipo')
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# Crear una copia explícita del subconjunto
df_año_points = vinos_csv_nuevo[['points', 'title']].copy()

# Extraer los años de la columna 'title'
df_año_points['año'] = df_año_points['title'].str.extract(r'\b(1[89]\d{2}|20\d{2})\b')

# Convertir la columna 'año' a float y agrupar por índice para obtener el máximo
df_año_points['año'] = df_año_points['año'].astype(float).groupby(df_año_points.index).max()

# Agrupar por 'año' y calcular el promedio de 'points', luego ordenar por puntaje promedio
df_año_points = df_año_points.groupby('año')['points'].mean().sort_values(ascending=False).reset_index().round(2)

# Ordenar los resultados por 'año' en orden ascendente
df_año_points = df_año_points.sort_values(by='año', ascending=True)

# Mostrar el resultado
print('Años y su relación con el promedio de puntos obtenidos:\n')
print(df_año_points)

print('\t\t\t\t\t\t\t\t--//--\n')

# Calcular la correlación entre el año de producción y el puntaje
correlacion = df_año_points[['año', 'points']].corr().iloc[0, 1]
print(f'Correlación entre el año de producción y el puntaje: {correlacion:.2f}\n')

# Identificar tipo de correlación
if correlacion == 0:
    print('No existe ninguna relación lineal entre el año y el puntaje')
elif abs(correlacion - 0) < abs(correlacion - 1) and abs(correlacion - 0) < abs(correlacion + 1):
    print('No hay una relación lineal clara entre el año y el puntaje')
elif abs(correlacion - 1) < abs(correlacion - 0) and abs(correlacion - 1) < abs(correlacion + 1):
    print('Existe una relación positiva fuerte entre el año y el puntaje')
elif abs(correlacion + 1) < abs(correlacion - 0) and abs(correlacion + 1) < abs(correlacion - 1):
    print('Existe una relación negativa fuerte entre el año y el puntaje')

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Relación entre el año de producción y el puntaje - Gráfico de regresión\n')

# Relación entre el año de producción y el puntaje - Gráfico
plt.figure(figsize=(12, 6))
sns.regplot(x='año', y='points', data=df_año_points, ci=100, marker="o", color=".3", line_kws=dict(color="r"))
plt.title('Relación entre el año de producción y el puntaje')
plt.xlabel('Año de producción')
plt.ylabel('points')
plt.grid(linestyle='--')
plt.show()

print('\t\t\t\t\t\t\t\t--//--\n')

# Creación de un DataFrame mean_points_x_province para realizar los calculos de country, province y points
mean_points_x_province = vinos_csv_nuevo[['country', 'province', 'points']]

# Calcular el puntaje promedio y la desviación estándar de puntos por país y provincia
df_mean_points_x_province = mean_points_x_province.groupby(['country', 'province'])['points'].agg(['mean', 'std']).round(2)

# Renombrar las columnas para mayor claridad
df_mean_points_x_province.columns = ['mean_points', 'std_points']

# Transformación de un nuevo DataFrame
df_mean_points_x_province = df_mean_points_x_province.reset_index()

# Mostrar los resultados
print('Variación de puntos entre provincias:\n')
print(df_mean_points_x_province)

print('\t\t\t\t\t\t\t\t--//--\n')
print('Gráfico - Puntos medios vs. desviación estándar por provincia y país - Gráfico de Burbuja\n')

# Puntos medios vs. desviación estándar por provincia y país - Gráfico
plt.figure(figsize=(14, 8))
scatter_plot = sns.scatterplot(data=df_mean_points_x_province, x='mean_points', y='std_points', hue='country', style='province')
plt.title('Puntos medios vs. desviación estándar por provincia y país')
plt.xlabel('mean_points')
plt.ylabel('std_points')
plt.grid(linestyle='--')
plt.tight_layout()

# Ajustar la leyenda para que solo muestre los países y esté fuera del gráfico
handles, labels = scatter_plot.get_legend_handles_labels()
country_handles = handles[:len(df_mean_points_x_province['country'].unique())]
country_labels = labels[:len(df_mean_points_x_province['country'].unique())]
legend = scatter_plot.legend(handles=country_handles, labels=country_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

print('\t\t\t\t\t\t\t\t--/FIN DEL PROGRAMA/--\n')

print("Presiona 'Esc' para cerrar el programa")

# Espera hasta que se presione la tecla 'Esc'
keyboard.wait('esc')

print("Programa cerrado")