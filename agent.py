import datetime
import json
import os

from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
import ast
import re
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain_openai import OpenAIEmbeddings
import streamlit as st

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = st.secrets['google']['application_credentials']
os.environ['OPENAI_API_KEY'] = st.secrets['openai']['api_key']

llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

project_id = 'cricket-analysis-443002' #Google Cloud ProjectID
dataset_id = 'cricksheet_1' #Google Cloud Bigquery dataset-id

# Create a SQLDatabase instance
bigquery_db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")
db=bigquery_db

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

assert len(prompt_template.messages) == 1
prompt_template.messages[0].pretty_print()
system_message = prompt_template.format(dialect="bigquery", top_k=5)
# Dictionary to manage multiple vector stores
vector_stores = {}


def create_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Example model
    vector_store = InMemoryVectorStore(embedding=embeddings)
    return vector_store

def add_to_vector_store(store_name, data):
    if store_name not in vector_stores:
        vector_stores[store_name] = create_vector_store()
    vector_stores[store_name].add_texts(data)

def query_as_list(db, query):
    print("extracting data")
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


players = query_as_list(db, "SELECT DISTINCT player_name FROM Players")
teams = query_as_list(db, "SELECT DISTINCT team_name FROM Players")
events = query_as_list(db, "SELECT DISTINCT event_name FROM Events")
officials = query_as_list(db, "SELECT DISTINCT official_name FROM Officials")
match_types = query_as_list(db, "SELECT DISTINCT match_type FROM Game_meta")

text_data = {'players':players,'teams':teams,'events':events,'officials':officials,'match_types':match_types }
retriever = {}
description = (
    "Use to look up values to filter on. Input is an approximate spelling "
    "of the proper noun, output is valid proper nouns. Use the noun most "
    "similar to the search."
)
for k,v in text_data.items():
    print (f"creating tool vector store for {k}")
    add_to_vector_store(k,v)
    retriever[k] = vector_stores[k].as_retriever(search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever[k],
        name=f"search_proper_nouns_{k}",
        description=description,
    )
    tools.append(retriever_tool)
    print(f"tool added for {k}")

# Add to system message
table_description = (
    "Description of each table is given below:"
    "1: Players: Details about players and their statics in a match.Use this stable for getting overall statitics of a player in a match"
    "2. Deliveries: Table contains information about deliveries. Each row has information about single delivery in a match"
    "3. Events: contain event information"
    "4. Game_meta: Overall information of a game"
    "5. Officials: details about officials"
)
suffix = (
    "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
    "the filter value using the 'search_proper_nouns_{}' tool corresponding to noun field based on query."
    "players, teams, events or officials"
    " Do not try to "
    "guess at the proper name - use this function to find similar ones and use the one most relavant to query."
)

conditions = ("Follow below conditions while constructing query:"
              "1. Use colums 'match_start_date' 'match_end_date' in Game_meta table for queries related to dates"
              "2. In Deliveries table, runs_on_delivery_by_batter and runs_on_delivery are runs on a single delivery. You need to do suitable aggregation on this field to find total runs"
              "3. Don't use match_id field to infer year, event etc."
              f"4. Current date is: {datetime.datetime.today()}. Use this date as reference for date related queries."
              "5. If you can't answer , apologise and ask user for more details"
              "6. After getting response, check if data is sufficient to properly answer user query. If not ")

system_msg = f"{system_message}\n\n{table_description}\n\n{suffix} \n\n {conditions}"

# agent_executor = create_react_agent(llm, tools, state_modifier=system_msg)
agent_executor = create_react_agent(llm,tools,state_modifier=system_msg)

# def process_user_query(query):
#     msg = None
#     for step in agent_executor.stream(
#             {"messages": [{"role": "user", "content": query}]},
#             stream_mode="values"
#     ):
#         msg = step["messages"][-1]
#         step["messages"][-1].pretty_print()
#     return msg.content

# Title or description of your app
st.title("Run Query with Agent")

# Input field to get user query
user_query = st.text_input("Enter your query:")

# Run the agent and display the output
if st.button("Run Query"):
    if user_query:
        with st.spinner("Processing..."):
            try:
                msg = {"messages": [{"role": "user", "content": user_query}]}
                response = agent_executor.invoke(msg)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")
