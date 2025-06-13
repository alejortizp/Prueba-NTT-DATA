import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import tool
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Any
from pathlib import Path
import warnings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

@dataclass
class Config:
    """Configuración centralizada del agente de datos"""
    csv_file: str = "coffee_final.csv"
    model: str = "gpt-4o"
    temperature: float = 0.0
    output_dir: str = "plots"
    max_rows_display: int = 10000
    figure_size: tuple = (12, 8)
    supported_plot_types: List[str] = field(default_factory=lambda: ['line', 'bar', 'scatter', 'histogram', 'boxplot'])
    
    def __post_init__(self):
        """Validación post-inicialización"""
        Path(self.output_dir).mkdir(exist_ok=True)

class DataLoader:
    """Clase para manejar la carga y preprocesamiento de datos"""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV con validación robusta
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            DataFrame preprocesado
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            pd.errors.EmptyDataError: Si el archivo está vacío
            Exception: Para otros errores de carga
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo {file_path} no encontrado en {os.getcwd()}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Archivo cargado exitosamente: {len(df)} filas, {len(df.columns)} columnas")
            
            if df.empty:
                raise pd.errors.EmptyDataError("El archivo CSV está vacío")
                
            return DataLoader._preprocess_dataframe(df)
            
        except pd.errors.EmptyDataError:
            raise
        except Exception as e:
            logger.error(f"Error al cargar CSV: {e}")
            raise Exception(f"Error al cargar o procesar el archivo CSV: {e}")
    
    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa el DataFrame detectando automáticamente tipos de datos
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame preprocesado
        """
        df_processed = df.copy()
        
        # Detectar y convertir columnas de fecha
        date_columns = DataLoader._detect_date_columns(df_processed)
        for col in date_columns:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
            logger.info(f"Columna '{col}' convertida a datetime")
        
        # Detectar y convertir columnas numéricas
        numeric_columns = DataLoader._detect_numeric_columns(df_processed)
        for col in numeric_columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            # Llenar NaN con 0 solo en columnas que parecen de cantidad/ventas
            if any(keyword in col.lower() for keyword in ['venta', 'cantidad', 'precio', 'total', 'amount', 'sales']):
                df_processed[col] = df_processed[col].fillna(0)
            logger.info(f"Columna '{col}' convertida a numérica")
        
        # Eliminar filas completamente vacías
        df_processed = df_processed.dropna(how='all')
        
        logger.info(f"Preprocesamiento completado: {len(df_processed)} filas restantes")
        return df_processed
    
    @staticmethod
    def _detect_date_columns(df: pd.DataFrame) -> List[str]:
        """Detecta columnas que probablemente contienen fechas"""
        date_keywords = ['fecha', 'date', 'time', 'dia', 'mes', 'año', 'year', 'month', 'day']
        date_columns = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                # Verificar si al menos el 50% de los valores no nulos parecen fechas
                sample = df[col].dropna().head(100)
                if not sample.empty:
                    try:
                        pd.to_datetime(sample, errors='coerce')
                        date_columns.append(col)
                    except:
                        pass
        
        return date_columns
    
    @staticmethod
    def _detect_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Detecta columnas que deberían ser numéricas"""
        numeric_columns = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Intentar convertir una muestra
                sample = df[col].dropna().head(100)
                if not sample.empty:
                    try:
                        pd.to_numeric(sample, errors='raise')
                        numeric_columns.append(col)
                    except:
                        pass
        
        return numeric_columns

class PlotGenerator:
    """Clase para generar visualizaciones"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @tool
    def create_and_save_plot(
        self,
        plot_type: str = 'line',
        x_col: str = '',
        y_col: str = '',
        hue_col: Optional[str] = None,
        title: str = 'Análisis de Datos',
        filters: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> str:
        """
        Genera y guarda una gráfica de los datos del DataFrame global.
        
        Args:
            plot_type: Tipo de gráfica ('line', 'bar', 'scatter', 'histogram', 'boxplot')
            x_col: Columna para el eje X
            y_col: Columna para el eje Y
            hue_col: Columna para diferenciar series
            title: Título de la gráfica
            filters: Filtros a aplicar (dict con columna: valor/lista_valores)
            
        Returns:
            Mensaje de éxito o error
        """
        global df_global
        
        if df_global is None:
            return "Error: DataFrame no cargado. No se puede generar la gráfica."
        
        # Validar tipo de gráfica
        if plot_type not in self.config.supported_plot_types:
            return f"Error: Tipo de gráfica '{plot_type}' no soportado. Tipos disponibles: {', '.join(self.config.supported_plot_types)}"
        
        try:
            df_filtered = self._apply_filters(df_global.copy(), filters or {})
            
            if df_filtered.empty:
                return "No hay datos para graficar después de aplicar los filtros."
            
            # Auto-detectar columnas si no se especifican
            x_col, y_col = self._auto_detect_columns(df_filtered, x_col, y_col, plot_type)
            
            # Validar columnas
            validation_result = self._validate_columns(df_filtered, x_col, y_col, hue_col)
            if validation_result != "OK":
                return validation_result
            
            # Generar gráfica
            return self._generate_plot(df_filtered, plot_type, x_col, y_col, hue_col, title)
            
        except Exception as e:
            logger.error(f"Error en create_and_save_plot: {e}")
            return f"Error al generar la gráfica: {e}"
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
        """Aplica filtros al DataFrame"""
        if not filters:
            return df
        
        if isinstance(filters, str):
            try:
                filters = json.loads(filters)
            except json.JSONDecodeError:
                logger.warning(f"No se pudo parsear filters como JSON: {filters}")
                return df
        
        for col, val in filters.items():
            if col not in df.columns:
                logger.warning(f"Columna de filtro '{col}' no encontrada")
                continue
            
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            elif pd.api.types.is_string_dtype(df[col]):
                df = df[df[col].str.contains(str(val), case=False, na=False)]
            else:
                df = df[df[col] == val]
        
        return df
    
    def _auto_detect_columns(self, df: pd.DataFrame, x_col: str, y_col: str, plot_type: str) -> tuple:
        """Auto-detecta columnas apropiadas si no se especifican"""
        if not x_col:
            # Buscar columna de fecha primero
            date_cols = [col for col in df.columns if df[col].dtype.name.startswith('datetime')]
            if date_cols:
                x_col = date_cols[0]
            else:
                # Buscar columnas categóricas
                categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
                if categorical_cols:
                    x_col = categorical_cols[0]
                else:
                    x_col = df.columns[0]
        
        if not y_col and plot_type != 'histogram':
            # Buscar columnas numéricas
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != x_col]
            if numeric_cols:
                # Priorizar columnas con palabras clave
                priority_keywords = ['venta', 'cantidad', 'precio', 'total', 'sales', 'amount', 'value']
                for keyword in priority_keywords:
                    priority_cols = [col for col in numeric_cols if keyword in col.lower()]
                    if priority_cols:
                        y_col = priority_cols[0]
                        break
                if not y_col:
                    y_col = numeric_cols[0]
        
        return x_col, y_col
    
    def _validate_columns(self, df: pd.DataFrame, x_col: str, y_col: str, hue_col: Optional[str]) -> str:
        """Valida que las columnas existan y sean apropiadas"""
        required_cols = [col for col in [x_col, y_col, hue_col] if col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            available_cols = list(df.columns)
            return f"Error: Columnas no encontradas: {missing_cols}. Columnas disponibles: {available_cols}"
        
        return "OK"
    
    def _generate_plot(self, df: pd.DataFrame, plot_type: str, x_col: str, y_col: str, hue_col: Optional[str], title: str) -> str:
        """Genera la gráfica específica"""
        plt.figure(figsize=self.config.figure_size)
        
        try:
            if plot_type == 'line':
                sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
            elif plot_type == 'bar':
                if hue_col:
                    sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, estimator='sum', errorbar=None)
                else:
                    grouped_data = df.groupby(x_col)[y_col].sum().reset_index()
                    sns.barplot(data=grouped_data, x=x_col, y=y_col)
            elif plot_type == 'scatter':
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col)
            elif plot_type == 'histogram':
                plt.hist(df[x_col].dropna(), bins=30, alpha=0.7)
                plt.xlabel(x_col)
                plt.ylabel('Frecuencia')
            elif plot_type == 'boxplot':
                sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col)
            
            plt.title(title)
            if plot_type != 'histogram':
                plt.xlabel(x_col)
                plt.ylabel(y_col)
            
            # Rotar etiquetas del eje X si son muchas o muy largas
            if df[x_col].nunique() > 10 or any(len(str(x)) > 10 for x in df[x_col].unique()[:10]):
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Guardar archivo
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).replace(' ', '_')[:50]
            file_name = f"{plot_type}_{safe_title}_{timestamp}.png"
            file_path = os.path.join(self.config.output_dir, file_name)
            
            plt.savefig(file_path, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            
            logger.info(f"Gráfica guardada: {file_path}")
            return f"Gráfica {plot_type} guardada exitosamente en: {file_path}"
            
        except Exception as e:
            plt.close()
            logger.error(f"Error al generar gráfica: {e}")
            return f"Error al generar la gráfica: {e}"

class DataAgent:
    """Clase principal del agente de análisis de datos"""
    
    def __init__(self, config: Config):
        self.config = config
        self.plot_generator = PlotGenerator(config)
        self.df = None
        self.agent = None
        
        # Validar API key
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")
        
        # Inicializar LLM
        self.llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature
        )
        
        logger.info("DataAgent inicializado correctamente")
    
    def load_data(self, file_path: Optional[str] = None) -> None:
        """Carga los datos y configura el agente"""
        global df_global
        
        file_path = file_path or self.config.csv_file
        
        try:
            self.df = DataLoader.load_data(file_path)
            df_global = self.df  # Para compatibilidad con la herramienta
            
            logger.info(f"Datos cargados: {len(self.df)} filas, {len(self.df.columns)} columnas")
            logger.info(f"Columnas: {list(self.df.columns)}")
            
            self._setup_agent()
            
        except Exception as e:
            logger.error(f"Error al cargar datos: {e}")
            raise
    
    def _setup_agent(self) -> None:
        """Configura el agente de Pandas DataFrame"""
        tools = [self.plot_generator.create_and_save_plot]
        
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            agent_type="tool-calling",
            allow_dangerous_code=False,  # Más seguro para producción
            extra_tools=tools,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
        logger.info("Agente configurado exitosamente")
    
    def query(self, question: str) -> str:
        """
        Procesa una consulta del usuario
        
        Args:
            question: Pregunta o solicitud del usuario
            
        Returns:
            Respuesta del agente
        """
        if not self.agent:
            return "Error: Agente no inicializado. Carga los datos primero."
        
        enhanced_prompt = f"""
        Analiza la siguiente consulta del usuario sobre el dataset. 

        Información del dataset:
        - Filas: {len(self.df)}
        - Columnas: {list(self.df.columns)}
        - Tipos de datos: {dict(self.df.dtypes)}

        Si la consulta solicita una visualización (palabras clave: gráfica, gráfico, plot, visualiza, muestra, compara visualmente, etc.):
        1. Usa la herramienta 'create_and_save_plot'
        2. Infiere los parámetros apropiados basándote en los datos y la consulta
        3. Los tipos de gráfica disponibles son: {', '.join(self.config.supported_plot_types)}

        Consulta del usuario: {question}
        """
        
        try:
            response = self.agent.invoke({"input": enhanced_prompt})
            return response.get("output", "No se pudo generar respuesta")
        
        except Exception as e:
            logger.error(f"Error procesando consulta: {e}")
            return f"Error al procesar la consulta: {e}"
    
    def get_data_summary(self) -> str:
        """Retorna un resumen de los datos cargados"""
        if self.df is None:
            return "No hay datos cargados"
        
        summary = f"""
        === RESUMEN DEL DATASET ===
        Filas: {len(self.df)}
        Columnas: {len(self.df.columns)}
        
        Columnas y tipos:
        {self.df.dtypes.to_string()}
        
        Primeras 5 filas:
        {self.df.head().to_string()}
        
        Información de valores nulos:
        {self.df.isnull().sum().to_string()}
        """
        
        return summary

def main():
    """Función principal mejorada"""
    try:
        # Inicializar configuración
        config = Config()
        
        # Crear agente
        agent = DataAgent(config)
        
        # Cargar datos
        agent.load_data()
        
        # Mostrar resumen
        print(agent.get_data_summary())
        
        print("\n🤖 ¡Agente de Análisis de Datos listo!")
        print("📊 Puedes hacer preguntas sobre los datos o solicitar visualizaciones")
        print("💡 Ejemplos:")
        print("   - '¿Cuáles son las ventas totales por país?'")
        print("   - 'Muestra un gráfico de ventas por mes'")
        print("   - 'Compara las ventas de Colombia y Brasil'")
        print("   - 'Haz un histograma de los precios'")
        print("📝 Escribe 'salir' para terminar\n")
        
        while True:
            try:
                query = input("🔍 Tu consulta: ").strip()
                
                if query.lower() in ['salir', 'exit', 'quit']:
                    print("👋 ¡Hasta luego!")
                    break
                
                if not query:
                    continue
                
                print("\n⏳ Procesando...")
                response = agent.query(query)
                print(f"\n🤖 Respuesta:\n{response}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                logger.error(f"Error en el bucle principal: {e}")
                print(f"❌ Error: {e}")
    
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        print(f"❌ Error crítico: {e}")

# Variable global para compatibilidad
df_global = None

if __name__ == "__main__":
    main()