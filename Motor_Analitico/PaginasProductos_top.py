"""
Importamos las librerías necesarias
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandasql import sqldf

#Para imprimir las url largas
pd.set_option('display.max_colwidth', None)

#Dataframe para 1_Data_Recordings.csv
df_1_Data_Recordings = pd.read_csv(r'Motor_Analitico\1_Data_Recordings.csv')

#Dataframe para 2_Data_Recordings.csv
df_2_Data_Metrics = pd.read_csv(r'Motor_Analitico\2_Data_Metrics.csv')

#Ejecutamos sql
pysqldf = lambda q: sqldf(q, globals())

#Unimos las paginas de entrada y las paginas de salida 
url_paginas = """
SELECT direccion_url_entrada AS pagina FROM df_1_Data_Recordings
UNION ALL
SELECT direccion_url_salida AS pagina FROM df_1_Data_Recordings
"""
paginas = pysqldf(url_paginas)

#Contamos el número de visitas por página
conteo_visitas = """
SELECT pagina, COUNT(*) AS total_visitas
FROM paginas
GROUP BY pagina
ORDER BY total_visitas DESC
"""

conteo_paginas = pysqldf(conteo_visitas)

#Reconocemos cual es el top 5 de las paginas
paginas_top5 = """
SELECT pagina, COUNT(*) AS total_visitas
FROM (
    SELECT direccion_url_entrada AS pagina 
    FROM df_1_Data_Recordings
    WHERE direccion_url_entrada != 'https://cloudlabslearning.com/'
    AND direccion_url_entrada NOT LIKE '%err=SUBSCRIPTION_NOT_FOUND%'
        AND direccion_url_salida NOT LIKE '%not-found%'
    
    UNION ALL
    
    SELECT direccion_url_salida AS pagina 
    FROM df_1_Data_Recordings
    WHERE direccion_url_salida != 'https://cloudlabslearning.com/'
    AND direccion_url_salida NOT LIKE '%err=SUBSCRIPTION_NOT_FOUND%'
        AND direccion_url_salida NOT LIKE '%not-found%'
)
GROUP BY pagina
ORDER BY total_visitas DESC
LIMIT 20
"""

top20 = pysqldf(paginas_top5)
top20.to_csv(r"Motor_Analitico\outputs\Top_20_paginas\top20_paginas.csv", index=False)
