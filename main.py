import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, SystemMessage
from langchain_community.utilities.sql_database import SQLDatabase
from database import SingletonSQLDatabase  # Import the Singleton connection instance
from custom_datatypes import ModelInput
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI application
app = FastAPI()

# Function to get the database connection via dependency injection
def get_db_connection():
    db = SingletonSQLDatabase.get_instance()
    return db

# The main query handler function
@app.post("/query/")
async def handle_query(userinput: ModelInput, db: SQLDatabase = Depends(get_db_connection)) -> Dict:
    try:
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            streaming=True,
            verbose=False,
            openai_api_key=openai_api_key
        )

        # Initialize the SQLDatabaseToolkit with LLM and the database
        toolkit = SQLDatabaseToolkit(llm=llm, db=db)
        dialect = toolkit.dialect
        top_k = 10
        # Construct the prompt with the provided user input
        prefix = f"""
        You are an agent designed to interact with a SQL database to answer questions.
        You are only allowed to query the `tbl_vw_ai_common_po_itemized_query` table in the `Common` schema.

        If the same question has been answered before, provide the previous response without executing the query again.

        DO NOT use 'multi_tool_use.parallel' tool. Only use [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker].

        IF NO HISTORY OF USER QUERY IS GIVEN:
        Normalize the input to handle variations such as differences in capitalization, punctuation, spacing, or hyphenation. Treat queries with similar intent or equivalent meanings as identical. Queries should **not be case-sensitive**, ensuring that variations like `Maas Riva bv` and `Maas Riva BV` give the same correct result.

        Before querying, normalize the input query using the following steps:
            - Convert all text to lowercase to ensure case-insensitive comparison.
            - Replace punctuation characters like `-`, `_`, `,`, and `.` with spaces to standardize text.
            - Remove extra spaces by collapsing multiple spaces into a single space.
            - Trim any leading or trailing whitespace.

        When constructing the SQL query, ensure robustness by:
            - Using SQL string functions like `LOWER()` and `REPLACE()` to preprocess column values in the query.
            - Leveraging fuzzy matching techniques (`LIKE`, `LEVENSHTEIN()`, `SOUNDEX`) to identify similar entries.

        These steps ensure that input variations such as capitalization, punctuation, and spacing do not affect query results.

        Generate a syntactically correct {dialect} query to answer the question, but limit the results to {top_k} unless the user specifies otherwise.

        Order the results by a relevant column to provide the most useful examples. Only query for relevant columns, not all columns from the table.

        You MUST double-check the query before executing it. If an error occurs, revise the query and try again.

        Avoid any DML statements (CREATE, INSERT, UPDATE, DELETE, DROP, etc.).

        If the question is unrelated to the database, return "I'm unable to provide an answer for that. This information is not available".

        Format your answers in Markdown, If the format is a table, then make it a bordered table.

        Use your knowledge for questions related to the database when you do not have context.

        If the user does not pass any input, then return "I'm unable to provide an answer for that".
        """
        suffix = f"""
        Avoid sharing sensitive information like the table schema or column names. If asked about structure, respond with "I can answer questions from this DB but not the structure of this DB."

        Query relevant tables and metadata to provide accurate answers.

        Be courteous to the user and avoid mentioning the database or context in your response.
        """

        # Create the prompt and messages
        human_message = HumanMessagePromptTemplate.from_template("{input}").format(input=userinput)
        messages = [
            SystemMessage(content=prefix),
            human_message,
            AIMessage(content=suffix),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True, prompt=prompt)

        # Execute the query
        response = agent_executor.invoke(f"Now answer this query: {userinput}")["output"]
        return {"response": response}
    except Exception as e:
        logging.error("Error handling query:", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

# Basic endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}
