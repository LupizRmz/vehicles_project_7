#!/usr/bin/env python
# coding: utf-8

# # ¿Qué vende un coche?
# 
# En este informe se presenta un anális de datos recopilados en los últimos años para determinar qué factores influyen en el precio de un vehículo. Estos datos provienen de los anuncios gratuitos de los vehiculos que se publican en el sitio web Crankshaft List.
# 
# Para este análisis, primero se examinan los datos para corregirlos en caso de ser necesario, esto es indagar en busca de valores atípicos y entradas mal escritas para generar datos limpios. Después se observan los parámetros principales y finalmente se obtiene la influencia de los parámetros en el precio.

# ## Inicialización

# In[1]:


# Cargar todas las librerías
import pandas as pd
import scipy.stats
import streamlit as st
import time


# ### Cargar datos

# In[4]:


# Cargar el archivo de datos en un DataFrame
car_data = pd.read_csv('../vehicles_us.csv')


# ### Explorar datos iniciales

# El dataset contiene los siguientes campos:
# - `price`
# - `model_year`
# - `model`
# - `condition`
# - `cylinders`
# - `fuel` — gasolina, diesel, etc.
# - `odometer` — el millaje del vehículo cuando el anuncio fue publicado
# - `transmission`
# - `paint_color`
# - `is_4wd` — si el vehículo tiene tracción a las 4 ruedas (tipo Booleano)
# - `date_posted` — la fecha en la que el anuncio fue publicado
# - `days_listed` — desde la publicación hasta que se elimina

# In[6]:


#información general/resumida sobre el DataFrame
car_data.info()


# In[7]:


#vista previa de los datos
car_data.head(5)


# In[8]:


car_data.describe()


# La información muestra valores ausentes y errores en el tipo de dato de la columna con fechas.

# ### Conclusiones y siguientes pasos
# 
# La información previa muestra que hay valores ausentes en las columnas 'model_year', 'cylinders', 'odometer', 'paint_color' y en 'is_4wd'. En esta ultima columna, faltan más de la mitad de los valores por lo que hay que analizar los valores de esta columna.
# 
# Los valores de la columna 'date_posted' no son de tipo fecha; hay que cambiar el tipo de dato
# 
# Hay que investigar los valores de las columnas, 'condition', 'fuel', 'transmission', 'type' y 'paint_color'  ya que pueden estar escritos de diferente forma refiriendose a lo mismo.

# ## Analizar valores de texto repetitivos
# 
# Se investigan los valores de las columnas 'is_4wd','model', 'condition', 'fuel', 'transmission', 'type' y 'paint_color'.

# In[10]:


for column in ['condition','fuel','transmission','type','paint_color','model']:
    print('valores en la columna',column,':')
    print(car_data[column].sort_values().unique())


# No se encuentran textos extraños en la mayoria de las columnas por lo que no se modifican valores con excepción de la columna 'model'. En esta última se estandarizan los valores para mejorar el análisis

# ## Tratar los valores ausentes y reducir los datos repetitivos

# Se corrigen datos repetitivos en columna 'model'

# In[13]:


#función para cambiar datos de columna 'model'
def replace_wrong_values(wrong_values, correct_value): 
    for wrong_value in wrong_values:
        car_data['model'] = car_data['model'].replace(wrong_value, correct_value) 


# In[14]:


#listas de agrupación
chevrolet =['chevrolet silverado 1500','chevrolet silverado 2500hd','chevrolet silverado 3500hd']
ford_150 = ['ford f150', 'ford f150 supercrew cab xlt']
ford_250 = ['ford f-250 sd','ford f-250 super duty','ford f250','ford f250 super duty']
ford_350 = ['ford f-350 sd','ford f350', 'ford f350 super duty']

#reemplazar valores de listas
replace_wrong_values(chevrolet,'chevrolet silverado')
replace_wrong_values(ford_150,'ford f-150')
replace_wrong_values(ford_250,'ford f-250')
replace_wrong_values(ford_350,'ford f-350')

#reemplazar valores únicos
car_data['model'] = car_data['model'].replace('chevrolet camaro lt coupe 2d', 'chevrolet camaro') 
car_data['model'] = car_data['model'].replace('ford focus se','ford focus')
car_data['model'] = car_data['model'].replace('ford fusion se','ford fusion')
car_data['model'] = car_data['model'].replace('ford mustang gt coupe 2d','ford mustang')
car_data['model'] = car_data['model'].replace('honda civic lx','honda civic')
car_data['model'] = car_data['model'].replace('jeep wrangler unlimited','jeep wrangler')
car_data['model'] = car_data['model'].replace('nissan frontier crew cab sv','nissan frontier')
car_data['model'] = car_data['model'].replace('jeep grand cherokee laredo','jeep grand cherokee')
car_data['model'] = car_data['model'].replace('toyota camry le','toyota camry')


# Se analizan las columnas con valores ausentes y se obtienen los porcentajes a modificar

# In[16]:


total_values = 51525

for column in ['is_4wd','model_year','cylinders','odometer']:
    print('valores en la columna',column,':')
    print(car_data[column].sort_values().unique())

    isna_sum = car_data[column].isna().sum()
    print('El porcentaje de valores ausentes en',column,'es de ', 
          round(isna_sum/total_values*100,2),'%')
    print()


print('El porcentaje de valores ausentes en paint_color es de ', 
      round( car_data['paint_color'].isna().sum()/total_values*100,2),'%')


# El porcentaje de valores ausentes por columna es significativamente alto por lo cual no se pueden eliminar todas estas filas.
# 
# La columna 'is_4wd' unicamente tiene dos valores que son 1 (True) y Nan; por lo que se asume que nan = 0 ó False.
# 
# En la columna 'paint_color' se establecen los datos desconocidos como 'unknown'
# 
# En las demás columnas no se observan datos extraños.

# In[17]:


#Cambiar Nan a 0
car_data['is_4wd'] = car_data['is_4wd'].fillna(0)

#Cambiar valores ausentes a 'unknown'
car_data['paint_color'] = car_data['paint_color'].fillna('unknown')


# En la columna 'odometer', se establece como valor representativo la mediana de todas las filas de cada año para llenar los valores ausentes.
# 
# En la columna'cylinders' se establece como valor representativo la moda de todas las filas de cada tipo de vehiculo.
# 
# En la columna 'model_year' se establecen como valor representativo la moda de todas las filas de cierto rango de millaje.

# In[19]:


#Filas donde se desconoce 'model_year' y 'odometer'
useless_rows = car_data[car_data['model_year'].isna() & car_data['odometer'].isna()]['price'].count()
print('Porcentaje de filas sin datos de año de modelo y millaje: ',round(useless_rows/total_values*100,2),'%')


# Los anuncios donde no se sabe el año del modelo ni el millaje no brindan información apta para el analisis. Los datos de 'model_year' se usaran como referencia para llenar 'odometer' y viceversa; asimismo, es un porcentaje bajo de datos por lo que se eliminan.

# In[21]:


car_data.drop(car_data[car_data['model_year'].isna() & car_data['odometer'].isna()].index,inplace=True)


# In[22]:


car_data['odometer'].describe()


# In[23]:


#escribir función que obtenga el rango de millaje
def odom_range(odometer):
    if odometer == 0: return 'odom = 0'
    elif odometer <=50000:  return '0 < odom <= 50,000'
    elif odometer <=100000: return '50,000 < odom <= 100,000'
    elif odometer <=150000: return '100,000 < odom <= 150,000'
    elif odometer <=200000: return '150,000 < odom <= 200,000'
    elif odometer <=250000: return '200,000 < odom <= 250,000'
    elif odometer <=300000: return '250,000 < odom <= 300,000'
    elif odometer <=350000: return '300,000 < odom <= 350,000'
    elif odometer <=400000: return '350,000 < odom <= 400,000'
    elif odometer <=450000: return '400,000 < odom <= 450,000'
    elif odometer <=500000: return '450,000 < odom <= 500,000'
    elif odometer <=550000: return '500,000 < odom <= 550,000'
    elif odometer <=600000: return '550,000 < odom <= 600,000'
    elif odometer <=650000: return '600,000 < odom <= 650,000'
    elif odometer <=700000: return '650,000 < odom <= 700,000'
    elif odometer <=750000: return '700,000 < odom <= 750,000'
    elif odometer <=800000: return '750,000 < odom <= 800,000'
    elif odometer <=850000: return '800,000 < odom <= 850,000'
    elif odometer <=900000: return '850,000 < odom <= 900,000'
    else:                   return 'odom > 900,000'


#comprobar función
for i in [0,55000,125000,852345,951000]:
    print(odom_range(i))


# In[24]:


# Crear una nueva columna basada en la función
car_data ['odom_range'] = car_data ['odometer'].apply(odom_range)
#ver nueva columna
car_data.head()


# In[25]:


odom_median = car_data.groupby('model_year')['odometer'].transform('median') 
car_data['odometer'].fillna(odom_median, inplace=True) 

cyl_mode = car_data.groupby('type')['cylinders'].transform(lambda x: x.mode()[0])
car_data['cylinders'].fillna(cyl_mode, inplace=True) 

model_year_mode = car_data.groupby('odom_range')['model_year'].transform(lambda x: x.mode()[0])
car_data['model_year'].fillna(model_year_mode, inplace=True) 


# In[26]:


car_data.isna().sum()


# In[27]:


#buscar valores ausentes en columna odometer
car_data.query('odometer.isna()')


# In[28]:


#investigar fila
car_data.query('model_year == 1929')


# Unicamente hay una fila donde el vehiculo es un modelo del año 1929; no hay valores que se tomen como referencia, por lo que se elimina esta fila. 

# In[29]:


#eliminar fila
car_data = car_data.dropna(subset=['odometer'])


# In[30]:


#verificar cambios
car_data.info()
car_data.head(10)


# Se tiene un dataframe sin valores ausentes

# ## Corregir los tipos de datos

# Se modifica el tipo de datos de la columna 'date_posted'

# In[33]:


#Se modifica el tipo de la columna 'date_posted' a tipo datetime
car_data['date_posted'] = pd.to_datetime(car_data['date_posted'], format='%Y-%m-%d')


# In[34]:


#Se modifica el tipo de la columna 'is_4wd' a tipo booleano
car_data['is_4wd'] = car_data['is_4wd'].astype('bool')


# In[35]:


#verificar cambios
car_data.info()
car_data.head()


# ## Enriquecer datos

# Se agrega información tomada de la misma tabla para para facilitar el análisis.

# In[36]:


#se agregan valores de dia de la semana, mes y año para cuando se colocó el anuncio
car_data['day_of_week'] = car_data['date_posted'].dt.weekday
car_data['month'] = car_data['date_posted'].dt.month
car_data['year'] = car_data['date_posted'].dt.year


# In[37]:


#se agregan los años del vehículo cuando el anuncio se colocó
car_data['vehicle_age'] = car_data['year']-car_data['model_year']+1


# In[38]:


#Se obtiene millaje promedio por vehiculo
car_data['avg_mileage_per_year'] = round( car_data['odometer'] / car_data['vehicle_age'],2 )


# En la columna condition, se reemplazan los valores de cadena con una escala numérica:
# - nuevo ('new') = 5
# - como nuevo ('like new') = 4
# - excelente ('excellent') = 3
# - bien ('good') = 2
# - bastante ('fair') = 1
# - para rescate ('salvage') = 0

# In[39]:


#Reemplazar los valores de condición con algo que se pueda manipular más fácilmente
cond_dict = {'new':5, 'like new':4, 'excellent':3, 'good':2, 'fair':1, 'salvage':0}
car_data['condition'] = car_data['condition'].map(cond_dict)


# In[40]:


#verificar cambios
car_data.info()
car_data.head()


# ## Comprobar datos limpios

# Se muestran todos los datos para saber exactamente que se va a utilizar para el análisis.

# In[41]:


#información general/resumida sobre el DataFrame
car_data.info()


# In[42]:


#Muestra de datos
car_data.head(10)


# ## Estudiar parámetros principales
# 
# A continuación, se analizan los parámetros principales que son:
# - Precio
# - Los años del vehículo cuando el anuncio se colocó
# - Millaje
# - Número de cilindros
# - Estado

# In[44]:


#analizar histograma, boxplot y datos de los parámetros principales
plt.subplots(figsize=(20, 8))

columns = ['price','vehicle_age','odometer','cylinders','condition']

for i, column in enumerate(columns):
    if column =='vehicle_age':
        car_data.query('model_year>0').boxplot(column=column, ax=plt.subplot(2, len(columns), i + 1))
        car_data.query('model_year>0').hist(column, ax=plt.subplot(2, len(columns), len(columns) + i + 1))
    elif column =='cylinders':
        car_data.query('cylinders>0').boxplot(column=column, ax=plt.subplot(2, len(columns), i + 1))
        car_data.query('cylinders>0').hist(column, ax=plt.subplot(2, len(columns), len(columns) + i + 1))
    else:
        car_data.boxplot(column=column, ax=plt.subplot(2, len(columns), i + 1))
        car_data.hist(column, ax=plt.subplot(2, len(columns), len(columns) + i + 1))

    plt.xlim(0, max(car_data[column]))
    plt.xticks(rotation=60)

plt.show()


# Se observan múltiples valores atípicamente altos en las columnas 'price', 'vehicle_age' y 'odometer'.

# In[45]:


#Se analizan a fondo las columnas con valores atípicos
for column in ['price','vehicle_age','odometer']:
    print(column)
    print(car_data[column].describe())
    print()


# In[46]:


car_data.query('price==1')


# Se encuentran carros en excelente estado con precios menores a 10 cosa que no es normal

# ## Estudiar y tratar valores atípicos

# In[48]:


# Determina los límites inferiores para valores atípicos
price_lim_inf = int(car_data['price'].quantile(0.02))
print('límite inferior de precio: ',price_lim_inf)


# In[51]:


# Determinar los límites superiores para valores atípicos
price_lim_sup = int(car_data['price'].quantile(.95))
print('límite superior de precio: ',price_lim_sup)

v_age_lim_sup = int(car_data['vehicle_age'].quantile(.95))
print('límite superior de edad del vehículo: ',v_age_lim_sup)

odom_lim_sup = int(car_data['odometer'].quantile(.95))
print('límite superior de millaje: ',odom_lim_sup)


# In[53]:


#Se obtiene el porcentaje de valores atípicos en las columnas problemáticas
dict = {'price, fijando el límite inferior,':'price< @price_lim_inf',
        'price, fijando el límite superior,': 'price> @price_lim_sup',
        'vehicle_age, fijando el limite superior':'vehicle_age> @v_age_lim_sup',
       'odometer, fijando el límite superior,':'odometer>@odom_lim_sup'}

print('Porcentaje de valores atípicos:')
for key,value in dict.items():
    print('en la columna',key, 'es de',round(car_data.query(value)['price'].count() / total_values *100,2),'%')
    print()


# Los porcentajes de valores atípicos son menores al 5% por lo que se eliminan para mejorar la visualización de los datos.

# In[54]:


# Almacena los datos sin valores atípicos en un DataFrame separado
car_data_with_limits = car_data.query('@price_lim_inf < price < @price_lim_sup and vehicle_age < @v_age_lim_sup and odometer<@odom_lim_sup')


# ## Estudiar parámetros principales sin valores atípicos

# A continuación se muestran los datos filtrados sin valores típicos y los originales

# In[55]:


#analizar histograma y datos de los parámetros principales
plt.subplots(figsize=(20, 8))

columns = ['price','vehicle_age','odometer','cylinders','condition']

for i, column in enumerate(columns):
    if column == 'vehicle_age':
            car_data.query('model_year>0').hist(column=column, ax=plt.subplot(2, len(columns), i + 1))
            plt.xticks(rotation=60)
            car_data_with_limits.query('model_year>0').hist(column, ax=plt.subplot(2, len(columns), len(columns) + i + 1))
            plt.xticks(rotation=60)
    elif column =='cylinders':
        car_data.query('cylinders>0').hist(column=column, ax=plt.subplot(2, len(columns), i + 1))
        plt.xticks(rotation=60)
        car_data_with_limits.query('cylinders>0').hist(column, ax=plt.subplot(2, len(columns), len(columns) + i + 1))
        plt.xticks(rotation=60)
    else:
        car_data.hist(column=column, ax=plt.subplot(2, len(columns), i + 1))
        plt.xticks(rotation=60)
        car_data_with_limits.hist(column, ax=plt.subplot(2, len(columns), len(columns) + i + 1))
        plt.xticks(rotation=60)

plt.show()


# Al analizar los histogramas de la columna 'price' con los datos originales, los valores atípicos altos dan un histograma con solo una columna muy alta; sin embargo al eliminarlos, se obtiene un histograma donde los precios mayormente son menores a 10,000 y conforme el precio sube, hay menores vehiculos de estos costos.
# 
# En la columna 'vechicle_age', con los datos origniales, hay un pico alto entre la edad de 0 a 25 y el resto de la gráfica esta aparentemente vacio por los pocos valores atípicos. Al limpiar los datos, se obtiene una distribución bimodal con un pico en 5 y otro en alrededor de 10.
# 
# La columna 'odometer' inicialmente mostraba un histograma limitado. Tras fijar el limite superior, se obtuvo un histograma con distribución normal la cual tiene un pico entre 100,000 y 150,000.
# 
# Los datos de las columnas 'cylinders' y 'condition' se mantienen iguales pues los datos originales no muestran valores atípicos que afecten el análisis.

# ## Periodo de colocación de los anuncios
# 
# A continuación de analizan los días en que los anuncios fueron mostrados (columna 'days_listed')

# In[56]:


plt.subplots(figsize=(10, 5))
car_data_with_limits.hist('days_listed',ax=plt.subplot(1,2,1))
plt.title('days listed')
car_data_with_limits.boxplot('days_listed',ax=plt.subplot(1,2,2))
plt.title('days listed')
plt.show()


# In[57]:


car_data_with_limits['days_listed'].describe()


# In[58]:


car_data_with_limits['days_listed'].median()


# Como se muestra en las gráficas y datos anteriores, en promedio los anuncios duran 39 días y se tiene una mediana de 33 días.
# 
# El periodo de colocación habitual de un anuncio esta entre 0 y 50 días.
# 
# A continuación se analizan los anuncios que se eliminaron rápidamente y los que son publicados por un tiempo anormalmente largo.

# In[59]:


for filter in ['condition', 'cylinders', 'fuel','transmission']:
    print(car_data_with_limits.query('days_listed>100').groupby(filter)['price'].count())
    print()


# La mayoría de los vehiculos que duraron más de 100 días anunciados tienen la siguientes características:
# - tipo de condición 2 y 3 (bien y excelente)
# - 4, 6 y 8 cilindros
# - tipo de combustible: gas
# - automáticos

# In[60]:


print('porcentaje de valores atípicos altos: ',round(car_data_with_limits.query('days_listed>100')['price'].count()/total_values*100,2),'%')


# In[61]:


plt.subplots(figsize=(10, 5))
car_data_with_limits.query('days_listed<100').hist('days_listed',ax=plt.subplot(1,2,1))
car_data_with_limits.query('days_listed<100').boxplot('days_listed',ax=plt.subplot(1,2,2))
plt.show()


# In[62]:


car_data_with_limits.query('days_listed<100')['days_listed'].describe()


# In[63]:


car_data_with_limits.query('days_listed<100')['days_listed'].median()


# Sin tomar en cuenta los valores atípicamente largos, en promedio los vehiculos duran anunciados un promedio de 36 días con una mediana de 32 días. 
# 
# En comparación con los datos anteriores, el promedio bajo 3 días y la mediana 1 día.

# ## Precio promedio por cada tipo de vehículo

# Se analiza el número de anuncios y el precio promedio de cada tipo de vehículo.

# In[64]:


car_data_with_limits.pivot_table(index = 'type',values='price',aggfunc=['count','mean'])


# A continuación, se muestra un gráfico con la cantidad de anuncios en cada tipo de vehículo.

# In[65]:


car_data_with_limits.groupby('type').count()['price'].sort_values(ascending=False).plot(kind='bar')
plt.show()


# Los 2 tipos de vehiculo con mayor número de anuncios son sedan y SUV.
# 
# Les siguen truck, pickup, entre otros con menor número de anuncios.

# ## Factores de precio

# A continuación se analizan distintos factores para ver su impacto en el precio. Para ello se toman como base los 2 tipos con mayor anuncios mostrados en el gráfico anterior que son sedan y SUV.
# 
# Los factores a tomar en cuenta son:
# - edad
# - millaje
# - condición
# - tipo de transmisión
# - color
# 
# Para las variables numéricas se crea una matriz de dispersión y para las categoricas graficos de caja.

# **Análisis de vehiculo tipo Sedan**

# In[67]:


# se analizan: edad, millaje y condición contra el precio del tipo sedan
car_data_with_limits_sedan = car_data_with_limits.query('type == "sedan"')
car_data_with_limits_sedan = car_data_with_limits_sedan.drop(['model_year', 'model', 'cylinders', 'fuel','is_4wd','date_posted',
                        'days_listed','odom_range', 'day_of_week', 'month','year','avg_mileage_per_year'], axis=1)

pd.plotting.scatter_matrix(car_data_with_limits_sedan,figsize=(8,5))
plt.show()


# In[69]:


numeric_columns = car_data_with_limits_sedan.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
print(correlation_matrix)


# En cuanto al tipo Sedan se observa que:
# - en la relación precio contra la condicion del vehiculo, mientras mejor la condición, más alto tiende a ser el precio.
# - en la relación precio contra millaje, a mayor millaje, menor es el precio.
# - en la relación precio contra edad del vehiculo, a mayor edad, menor es el precio.

# In[70]:


car_data_with_limits_sedan.groupby('transmission')['price'].count()


# Cada tipo de transmisión tiene más de 50 entradas o anuncios, por lo que esta muestra es válida para el análisis.

# In[71]:


car_data_with_limits_sedan.boxplot(column='price',by='transmission',figsize=(12,5))
plt.show()


# Los diagramas de caja anteriores muestran que, en cuanto al tipo de transmición, en promedio los precios son similares sin embargo los vehiculos automaticos tienden a subir más de precio que los demás.

# In[72]:


car_data_with_limits_sedan.groupby('paint_color')['price'].count()


# Los colores naranja, morado y amarillo no pueden tomarse como referencia porque son una muestra muy pequeña. No se toman en cuenta para el analisis del tipo sedan.

# In[73]:


car_data_with_limits_sedan.query('paint_color not in ["orange","purple","yellow"]').boxplot(column='price',by='paint_color',figsize=(12,5))
plt.show()


# In[74]:


car_data_with_limits_sedan.query('paint_color not in ["orange","purple","yellow"]').groupby('paint_color')['price'].mean().sort_values(ascending=False)


# In[75]:


car_data_with_limits_sedan.query('paint_color not in ["orange","purple","yellow"]').groupby('paint_color')['price'].median().sort_values(ascending=False)


# En cuanto a la relación del precio con el color del vehiculo, los colores con tendencia a precio alto son negro, personalizado, gris, rojo y blanco, que tienen el mismo rango; seguidos de los colores azul y plateado y finalmente, los de precio más bajo tienden a ser los cafes y verdes.

# **Análisis de vehiculo tipo SUV**

# In[76]:


# se analizan: edad, millaje y condición contra el precio del tipo SUV
car_data_with_limits_suv = car_data_with_limits.query('type == "SUV"')
car_data_with_limits_suv = car_data_with_limits_suv.drop(['model_year', 'model', 'cylinders', 'fuel','is_4wd','date_posted',
                        'days_listed', 'odom_range','day_of_week', 'month','year','avg_mileage_per_year'], axis=1)

pd.plotting.scatter_matrix(car_data_with_limits_suv,figsize=(8,5))
plt.show()


# In[78]:


numeric_columns_suv = car_data_with_limits_suv.select_dtypes(include=['number'])
correlation_matrix_suv = numeric_columns_suv.corr()
print(correlation_matrix_suv)


# En cuanto al tipo SUV se observa que:
# - en la relación precio contra la condicion del vehiculo, mientras mejor la condición, más alto tiende a ser el precio.
# - en la relación precio contra millaje, a mayor millaje, menor es el precio.
# - en la relación precio contra edad del vehiculo, a mayor edad, menor es el precio.

# In[79]:


car_data_with_limits_suv.groupby('transmission')['price'].count()


# In[80]:


car_data_with_limits_suv.boxplot(column='price',by='transmission',figsize=(12,5))
plt.show()


# En el caso de los vehiculos de tipo SUV, los límites superiores del precio son similares mientras que el límite inferior de los de transmisión manual es ligeramente mayor a los automáticos y ambos son menores a otros tipos de transmisión. Al comparar los promedios, sucede lo mismo, el menor es el de los SUVs automáticos, seguidos por los de transmición manual y los del promedio más alto son los otros dos.

# In[81]:


car_data_with_limits_suv.groupby('paint_color')['price'].count()


# Los colores morado y amarillo no pueden tomarse como referencia porque son una muestra muy pequeña. No se toman en cuenta para el analisis del tipo SUV.

# In[82]:


car_data_with_limits_suv.query('paint_color not in ["purple","yellow"]').boxplot(column='price',by='paint_color',figsize=(12,5))
plt.show()


# In[83]:


car_data_with_limits_suv.query('paint_color not in ["purple","yellow"]').groupby('paint_color')['price'].mean().sort_values(ascending=False)


# In[84]:


car_data_with_limits_suv.query('paint_color not in ["purple","yellow"]').groupby('paint_color')['price'].median().sort_values(ascending=False)


# Al analizar el precio de los vehiculos SUV tomando en cuenta el color, pareciera que el color negro tiende a ser el más caro; esto tomando en cuenta su limite superior así como su promedio. Sin embargo, al comparar los promedios y las medianas, el color naranja es el de mayor precio. A este color le sigue el negro, entre otros. Los de promedio y mediana más bajos son el verde, cafe y azul.

# ## Conclusión general
# 
# Previo a la realización de este análisis, se tuvieron que modificar valores, eliminar filas con datos incompletos; así como llenar filas con valores representativos. 
# 
# - Se estandarizaron datos como el tipo de modelo del vehiculo ya que parece ser una entrada de datos manual y habia filas refiriendose al mismo vechiculo escrito de diferente forma. 
# 
# En cuanto a los valores ausentes:
# - En la columna 'is_4wd' todos los valores existentes estaban marcados como 1, que representa un True, por lo que se fijaron los ausentes en 0 para representar un False.
# 
# - En las columnas 'model_year' y 'cylinders' el 7.02 % y 10.21% de las filas, respectivamente, tenían datos ausentes. Para no eliminar las filas con otros parámetros importantes a considerar, se fijaron estos valores en cero.
# 
# - En la columna 'paint_color', se fijaron los valores ausentes, los cuales representan el 17.99% de los datos, como 'unknown' (desconocido). 
# 
# - En la columna 'odometer' se optó por usar la mediana de cada decada como un valor representativo para las filas con valores ausentes. 
# 
# En cuanto a los valores atípicos, las columnas 'price', 'vehicle_age' y 'odometer' tenían datos altos, los cuales no se tomaron en cuenta para el análisis. Asimismo, a la columna 'price' se le fijo un límite inferior pues había filas con precios extañamente bajos.

# **Resultado del análisis**
# Tomando como referencia los vehiculos de tipo Sedan y SUV se puede afirmar que, como es de esperarse, el precio de un vehiculo
# - tiende a ser más alto mientras mejor sea su condición (coeficientes de 0.31 y 0.26)
# - tiende a ser más bajo mientras mayor sea su millaje (coeficientes de -0.62 y -0.59)
# - tiende a ser más bajo mientras mayor sea su edad (coeficientes de -0.69 y -0.66)
# 
# Los coeficientes de correlación en ambos tipos de vehiculo muestran que el orden de influencia de estos 3 factores es:
# 1. edad del vehiculo
# 2. millaje del vehiculo
# 3. condición del vehiculo
# 
# 
# Al comparar los tipos de transmisión en los vehiculos de tipo sedan, al tomar en cuenta los valores atípicos, los automaticos tienden a ser los más caros. Estos y los de otro tipo de transmisión tienen rangos de precio similares mientras que los manuales son los de un rango más barato.
# 
# A diferenicia de esto, al analizar los de tipo SUV, los de menor precio tienden a ser los automáticos, seguidos de los manuales y los más caros estan clasificados como otros.
# 
# Al comparar los precios tomando en cuenta los colores, en ambos vehiculos el color negro tiende a ser más caro y los más baratos son los verdes y cafes. 
# Entre los precios altos del sedan se encuentran los colores negro, personalizado, gris, rojo y blanco, seguidos de azul y plateado y finalmente, los de precio más bajo tienden a ser los cafes y verdes.
# Entre los precios altos del SUV estan el negro y naranja y los más bajos son el verde café y azul.
