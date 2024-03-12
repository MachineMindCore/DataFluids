# DataFluids (Actualizacion)

![Inicio de DataFluids](/images/home.png)

El proyecto DataFluids es un conglomerado de herramientas de analisis de datos para el contexto de mercado electrico en el pais (Colombia), presentadas en formato de dashboard en el curso "Introduccion a los Sistemas de Energia Electrica" 2020-2S. La version actual esta reducida solo como demostracion del componente estadistico y actualizacion de librerias y funcionalidades.

# Funcionamiento

El dashboard se basa en una aplicacion web generada por streamlit y usa la base de datos del administrador de mercado XM a travez de la API implementada en pydataxm. Cualquier cambio en la distribucion y acumulacion de datos esta asociada a la API.

# Herramientas

Las herramientas estan divididas por secciones:

    * Agentes: Lista de agentes que participan en el mercado electrico y sus propiedades tecnologicas y energeticas.

    * Explorador: Exploracion de demanda electrica en diferentes zonas del pais.

    * Predictor: Usando los datos de XM construye modelos RandomForest inmediatos con 1 año de entrenamiento. Estas predicciones sirven para estimar la energia dia a dia durante todo el año.

    * Proyector: Proyecta datos de potencia para una distribucion solar o eolica configurable para todos los dias del año caracterizando por el mes.

# Contenido (Ejemplo)

## Agentes
![Lista de agentes.](/images/agents.png)

## Explorador
![Demanda de agente.](/images/explorer1.png)

![Precio nacional de bolsa.](/images/explorer2.png)

![Volumen util.](/images/explorer3.png)

## Predictor
![Prediccion de demanda.](/images/predictor1.png)

![Correlacion.](/images/predictor2.png)

## Proyector

![Proyeccion de potencia (Enero).](/images/proy_enero.png)

![Proyeccion de potencia (Junio).](/images/proy_junio.png)

![Proyeccion de potencia (Diciembre).](/images/proy_dic.png)

## Instalacion

Asegurese de que la instalacion de los paquetes sea dentro de un _virtual enviroment_.

´´´bash

git clone https://github.com/MachineMindCore/DataFluids.git
cd DataFluids
pip install -r requirements.txt

´´´

Una vez esten listos los requisitos solo basta con ejecutar el cliente de streamlit.

´´´bash

streamlit run main.py

´´´