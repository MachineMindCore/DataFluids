# DataFluids (Actualización)

![Inicio de DataFluids](/images/home.png)

El proyecto DataFluids es un conglomerado de herramientas de análisis de datos para el contexto de mercado eléctrico en el país (Colombia), presentadas en formato de dashboard en el curso "Introducción a los Sistemas de Energía Eléctrica" 2020-2S. La versión actual está reducida solo como demostración del componente estadístico y actualización de librerías y funcionalidades.

## Funcionamiento

El dashboard se basa en una aplicación web generada por Streamlit y usa la base de datos del administrador de mercado XM a través de la API implementada en pydataxm. Cualquier cambio en la distribución y acumulación de datos está asociado a la API.

## Herramientas

Las herramientas están divididas por secciones:

- Agentes: Lista de agentes que participan en el mercado eléctrico y sus propiedades tecnológicas y energéticas.
- Explorador: Exploración de demanda eléctrica en diferentes zonas del país.
- Predictor: Usando los datos de XM construye modelos RandomForest inmediatos con 1 año de entrenamiento. Estas predicciones sirven para estimar la energía día a día durante todo el año.
- Proyector: Proyecta datos de potencia para una distribución solar o eólica configurable para todos los días del año caracterizando por el mes.

## Contenido (Ejemplo)

### Agentes

![Lista de agentes.](/images/agents.png)

### Explorador

![Demanda de agente.](/images/explorer1.png)

![Precio nacional de bolsa.](/images/explorer2.png)

![Volumen util.](/images/explorer3.png)

### Predictor

![Predicción de demanda.](/images/predictor1.png)

![Correlación.](/images/predictor2.png)

### Proyector

![Proyección de potencia (Enero).](/images/proy_enero.png)

![Proyección de potencia (Junio).](/images/proy_junio.png)

![Proyección de potencia (Diciembre).](/images/proy_dic.png)

## Instalación

Asegúrese de que la instalación de los paquetes sea dentro de un _virtual environment_.

\```bash
git clone https://github.com/MachineMindCore/DataFluids.git
cd DataFluids
pip install -r requirements.txt
\```

Una vez estén listos los requisitos solo basta con ejecutar el cliente de Streamlit.

\```bash
streamlit run main.py
\```
