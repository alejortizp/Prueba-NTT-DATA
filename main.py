#import os
import streamlit as st
import pyodbc
import os
import boto3
from tools import get_schema_info, create_database_conecction, text, tool, type_chart_router
from dotenv import load_dotenv
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Annotated, Optional, Literal, Dict, Any, List, Callable, Sequence
from langgraph.graph.message import AnyMessage, add_messages
from langchain_groq import ChatGroq
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from uuid import uuid4

###################### Cargar variables de entorno ######################
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = "us.anthropic.claude-3-5-haiku-20241022-v1:0" # "llama-3.3-70b-versatile"
client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
llm = ChatBedrock(model_id = model, client=client, model_kwargs={'temperature':0.1,'top_p':0.4})

###################### Clase state que utiliza Laggraph ######################
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    current_user: str
    attempts: int
    relevance: str
    sql_error: bool
    type_chart: str
    response_ia: str

def execute_sql(state: AgentState) -> str:
    """Este tool realiza la ejecucion de una consulta SQL proporcionada por un LLM y devuelve los datos en forma de json para ser presentados por otro LLM.    
    Args:
        state: estados de AgentState
    Returns:
        state: retorna la modificacion de los state de AgentState    
    """   
    sql_query = state["sql_query"].strip()
    session = create_database_conecction("session")()# SessionLocal()
    
    try:        
        if sql_query.lower().startswith("select"):
            result = session.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            if rows:
                header = ",".join(columns)
                state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                data = "; ".join([" | ".join(str(row.get(col, "")) for col in columns) for row in state["query_rows"]])    
                formatted_result = f"{header}\n{data}"
            else:
                state["query_rows"] = []
                formatted_result = "Resultados no encontrados."            
            state["query_result"] = formatted_result
            state["sql_error"] = False            
        else:
            session.commit()
            state["query_result"] = "La acción se ha completado con éxito."
            state["sql_error"] = False
    except Exception as e:
        state["query_result"] = f" > ERROR!! -> Error al ejecutar la consulta SQL: {str(e)}"
        state["sql_error"] = True        
    finally:
        session.close()
    return state

###################### Nodo para validar la relevancia del prompt ######################
class CheckRelevance(BaseModel):
    relevance: str = Field(description="Indica si la pregunta está relacionada con el esquema de la base de datos. 'relevante' o 'no_relevante'.")

def check_relevance(state: AgentState, config: RunnableConfig) -> str:
    "Valida la relevacia que tiene la pregunta en cuanto al esquema de la base de datos proporcionado"    
    
    messages = state["messages"]   
    last_messages = messages[-7:] if messages else []
    formatted_messages = "\n".join(
            f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
            for msg in last_messages
        )    
    question = state["question"]
    schema = get_schema_info(create_database_conecction("engine"))
    
    system = """Eres un asistente que determina si una pregunta determinada está relacionada con el siguiente esquema de base de datos.
Esquema:
{schema}
Interacciones anteriores:
{formatted_messages}
Responde sólo con "relevante" o "no_relevante".
""".format(schema=schema, formatted_messages=formatted_messages)
    
    human = f"pregunta: {question}"
    
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    #llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0)   #"llama3-70b-8192"     
    structured_llm = llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke(
                                         {
                                          "question": state["question"], 
                                          "attempts": state["attempts"]
                                         }
                                        )
    state["relevance"] = relevance.relevance    
    
    return state

###################### Nodo para convertir la pregunta natural a sql ######################
class ConvertToSQL(BaseModel):
    sql_query: str = Field(description="La consulta SQL correspondiente a la pregunta en lenguaje natural del usuario.")

def convert_nl_to_sql(state: AgentState, config: RunnableConfig) -> str:
    "Convierte la pregunta natural a "    
    question = state["question"]
    messages = state["messages"]   
    last_messages = messages[-7:] if messages else []
    formatted_messages = "\n".join(
            f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
            for msg in last_messages
        )

    schema = get_schema_info(create_database_conecction("engine"))

    system = """Eres un asistente que convierte preguntas en lenguaje natural en consultas SQL basadas en el siguiente esquema y historial de interacciones:
Esquema:
{schema}
Estas son las interacciones anteriores:
{formatted_messages}

Si te solicitan una serie temporal convierte las fechas de acuerdo a lo siguiente:
    * Si la serie temporal es por meses convierte "FORMAT( Convert(date,id_fecha_desembolso),'yyyy-MM')" la fecha a mes año, ejemplo: junio 2025
    * Si la serie temporal es por dias convierte la fecha a yyyy-mm-ddd, ejemplo: 2025-06-01
El formato de fechas que debes manegar es YYYYMMDD para id_fecha
Cuanto te pidan informacion sobre meses completos siempre usa el formato con like 'YYYYMM%'
Maneja sintaxis solamente de SQL server o transact sql.
Cuando sean setencias de agregacion count, sum, min, max, avg, etc, asignale alias a las columnas como mejor consideres
Cuando te pidan datos puntuales solo consulta los unicos, por ejemplo: nombre de la identificacion 1111111

Proporciona solo la consulta SQL sin explicaciones.
""".format(schema=schema, 
           formatted_messages=formatted_messages)
    
    convert_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Pregunta: {question}"),
        ]
    )
    
    #llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0)#llama-3.3-70b-versatile
    structured_llm = llm.with_structured_output(ConvertToSQL)  
    sql_generator = convert_prompt | structured_llm
    result = sql_generator.invoke(
                                  {
                                    "question": question
                                  }
                                 )
    state["sql_query"] = result.sql_query
    state["messages"].append(AIMessage(content=result.sql_query))
    
    return state

###################### Nodo para ejecutar consulta sql entregada por el llm ######################

###################### Nodo para ejecutar consulta sql entregada por el llm ######################
def generate_human_readable_answer(state: AgentState):
      
    messages = state["messages"]
    last_messages = messages[-7:] if messages else []

    formatted_messages = "\n".join(
            f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
            for msg in last_messages
        )
    
    sql = state["sql_query"]    
    result = state["query_result"]    
    query_rows = state.get("query_rows", [])
    sql_error = state.get("sql_error", False)   
    type_chart = state["type_chart"]
    
    system = f"""Eres un asistente que convierte los resultados de consultas SQL en respuestas claras y en lenguaje natural, teniendo en cuenta el historial de interacciones.    
Estas son las interacciones anteriores:
{formatted_messages}
"""
    
    if sql_error:        
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"""SQL Query:
{sql}
Resultado:
{result}
Formule un mensaje de error claro y comprensible en una sola oración informándoles sobre el problema."""
                ),
            ]
        )
    elif sql.lower().startswith("select"):
        if not query_rows:            
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",f"""SQL Query:
{sql}
Resultado:
{result}
Formule una respuesta clara y comprensible a la pregunta original en una sola oración, y mencione que no se encontraron los datos solicitados.
"""
                    ),
                ]
            )
        else:
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",f"""SQL Query:
{sql}
Resultado:
{result}
Formula una respuesta clara y comprensible a la pregunta original en una sola oración, se amable al responder.
Si la sentencia sql GROUP BY presentalo en forma de tabla.
Muestra la consulta SQL de forma ordenada en un bloque de codigo.
No muestres notas no expliques como lo hiciste."""
                    ),
                ]
            )
    else:        
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"""SQL Query:
{sql}
Resultado:
{result}
Formule un mensaje de confirmación claro y comprensible en una sola oración, confirmando que su solicitud se ha procesado correctamente.
Si la sentencia sql GROUP BY presentalo en forma de tabla
Muestra la consulta SQL de forma ordenada en un bloque de codigo.
No muestres notas no expliques como lo hiciste."""
                ),
            ]
        )        
    #llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0)#"llama3-70b-8192"
    human_response = generate_prompt | llm | StrOutputParser()
    answer = human_response.invoke({})
    state["response_ia"] = answer
    return state

###################### Nodo para regenerar la pregunta natural llm ######################
class RewrittenQuestion(BaseModel):
    question: str = Field(description="Pregunta reescrita por el llm.")
    
def regenerate_query(state: AgentState):
    question = state["question"]    
    system = """Eres un asistente que reformula una pregunta original para permitir consultas SQL más precisas. Asegúrate de que se conserven todos los detalles necesarios, como las uniones de tablas, para recuperar datos completos y precisos.
    """
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human","""pregunta original: {question}
Reformula la pregunta para permitir consultas SQL más precisas, garantizando que se conserven todos los detalles necesarios.""",
            ),
        ]
    )
    #llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0)#"llama-3.3-70b-versatile"
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm
    rewritten = rewriter.invoke({
                                    "question":question
                                })
    state["question"] = rewritten.question
    state["attempts"] += 1    
    return state

###################### Nodo para regenerar respuestas genericas ######################
def generate_generic_response(state: AgentState)-> dict:
    
    question = state["question"]
    messages = state["messages"]    
    
    last_messages = messages[-7:] if messages else []
            
    formatted_messages = "\n".join(
            f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
            for msg in last_messages
        )       

    system_prompt = f"""Eres un asistente que ayuda a generar consultas en una base de datos de cartera que responde de manera muy amable.
Si la pregunta realizada no fue relevante, entonces genera la respuesta a lo que te pregunte, puede ser cualquier indole.
    
Utiliza estas interacciones anteriores para mejorar tus respuestas:
{formatted_messages}
"""
    human_prompt = "Pregunta: {question}"
        
    funny_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )    
    #llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0)#"llama3-70b-8192"
    response_chain = funny_prompt | llm | StrOutputParser()    
    message = response_chain.invoke({
                                        "question":question,
                                        "messages": formatted_messages
                                    })
    state["response_ia"] = message
    state["type_chart"] = "grafico_no_relevante"
    state["messages"].append(AIMessage(content=message))
    return state

###################### Nodo para validar que tipo de graficos se puede generar ######################
def check_relevance_chart(state: AgentState):
    sql = state["sql_query"]     
    messages = state["messages"]    
    
    last_messages = messages[-7:] if messages else []
            
    formatted_messages = "\n".join(
            f"{'Usuario' if isinstance(msg, HumanMessage) else 'Asistente'}: {msg.content}"
            for msg in last_messages
        )  
    
    system_prompt = f"""Eres un asistente que va a definir cuantas variables tiene una consulta SQL. A partir de la informacion, recomendara tipos de graficos que se pueden implementar con plotly, ten encuenta las siguientes indicaciones:
- Analiza las interacciones para ver si el humano pide algun grafico en especifico.
- La consulta solo debe de ser con tipo Select, si no contiene select retorna "grafico_no_relevante"
- Las variables son las columnas de la consulta select SQL
- Si los datos son de series temporales recomienda este grafico:
    * grafico_serie_temporal
- Si es 1 variable no recomiendes nada y retorna la salida de "grafico_no_relevante"
- Si son 2 variables recomienda alguno de los siguientes tipos de grafico que mas se ajuste:
    * grafico_torta
    * grafico_barras
    * grafico_lineas
    * grafico_cajas
- Si son 3 o mas variables recomienda alguno de los siguientes tipos de grafico:
    * grafico_sunburst

Retorna el tipo grafico que recomiendas solamente.
No siemprese recomiendes lo mismo, mira el historial de interacciones para validarlo
No des ninguna explicacion ni notas al respecto.

Interacciones anteriores:
{formatted_messages}
"""
    human_prompt = "Consulta SQL: {sql_query}"
        
    chart_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )    
    #llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model, temperature=0)#"llama3-70b-8192"
    response_chain = chart_prompt | llm | StrOutputParser()    
    result = response_chain.invoke({
                                        "sql_query":sql,
                                        "formatted_messages":formatted_messages
                                    })
    state["type_chart"] = result
        
    return state

###################### Funciones para routear los nodos del grado ######################
def end_max_iterations(state: AgentState):
    state["query_result"] = "Intentalo de nuevo por favor."
    return state

def relevance_router(state: AgentState):
    if state["relevance"].lower() == "relevante":
        return "convert_to_sql"
    else:
        return "generate_generic_response"

def check_attempts_router(state: AgentState):
    if state["attempts"] < 3:
        return "convert_to_sql"
    else:
        return "end_max_iterations"

def execute_sql_router(state: AgentState):
    if not state.get("sql_error", False):
        return "check_relevance_chart"
    else:
        return "regenerate_query"

###################### Union de aristas de los nodos en el grafo ######################
workflow = StateGraph(AgentState)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("convert_to_sql", convert_nl_to_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("generate_generic_response", generate_generic_response)
workflow.add_node("end_max_iterations", end_max_iterations)
workflow.add_node("check_relevance_chart", check_relevance_chart)

workflow.add_conditional_edges(
    "check_relevance",
    relevance_router,
    {
        "convert_to_sql": "convert_to_sql",
        "generate_generic_response": "generate_generic_response",
    },
)

workflow.add_edge("convert_to_sql", "execute_sql")
workflow.add_conditional_edges(
    "execute_sql",
    execute_sql_router,
    {
        "check_relevance_chart": "check_relevance_chart",
        "regenerate_query": "regenerate_query",
    },
)

workflow.add_edge("check_relevance_chart","generate_human_readable_answer")
workflow.add_conditional_edges(
    "regenerate_query",
    check_attempts_router,
    {
        "convert_to_sql": "convert_to_sql",
        "max_iterations": "end_max_iterations",
    },
)

workflow.add_edge("generate_human_readable_answer", END)
workflow.add_edge("generate_generic_response", END)
workflow.add_edge("end_max_iterations", END)


workflow.set_entry_point("check_relevance")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

###################### Arrancar servicio de streamlit ######################
st.set_page_config(
    page_title = "Preguntas Cartera"
    )

st.title("Preguntas cartera Finanzauto")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hola, estoy acá para ayudarte a responder tus preguntas sobre los datos de Finanzauto!"),
    ]

if "ai_queries" not in st.session_state:
    st.session_state.ai_queries = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Escribe lo que deseas saber...")

if user_query:
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    result_1 = ""
    chart = None
    with st.chat_message("Human"):
        st.markdown(user_query.replace("$","\$"))
    
    with st.chat_message("AI"):
        
        try: 
            config = {"configurable": {"thread_id": st.session_state.conversation_id}}
            result = app.invoke({"question": user_query,
                                 "attempts": 0,
                                 "messages": st.session_state.chat_history.copy()
                                }, 
                                config)
            
            response = result["response_ia"]            
            type_chart = result["type_chart"]
            print(type_chart)
            if type_chart != "grafico_no_relevante":
                sql_query = result["sql_query"]            
                print(f"Resultados base {sql_query}")            
                chart = type_chart_router(type_chart,sql_query)
                
        
        except Exception as e:
            st.write(e)
        st.write(response.replace("$","\$"))
        if chart is not None:
            st.plotly_chart(chart)
    
    st.session_state.chat_history.append(AIMessage(content=response))