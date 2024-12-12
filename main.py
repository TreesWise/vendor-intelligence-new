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
from apscheduler.schedulers.background import BackgroundScheduler

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI application
app = FastAPI()

# Function to keep the database connection alive
def keep_connection_alive():
    try:
        db = SingletonSQLDatabase.get_instance()  # Get the singleton database instance
        db.run("SELECT 1")  # Execute a simple query to keep the connection alive
        logging.info("Database connection kept alive.")
    except Exception as e:
        logging.error("Error in keep_connection_alive:", exc_info=True)


# Initialize APScheduler
scheduler = BackgroundScheduler()

# Schedule the keep_connection_alive task to run every 10 seconds
scheduler.add_job(keep_connection_alive, 'interval', seconds=10)

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

        prefix = """
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
            - Leveraging fuzzy matching techniques (`LIKE`, `LEVENSHTEIN()`, `SOUNDEX`) to identify similar entries. Example : when a user makes a spelling mistake in vendor name.
        
        These steps ensure that input variations such as capitalization, punctuation, and spacing do not affect query results.
        
        Generate a syntactically correct {dialect} query to answer the question, but limit the results to {top_k} unless the user specifies otherwise.
        
        Order the results by a relevant column to provide the most useful examples. Only query for relevant columns, not all columns from the table.
        
        You MUST double-check the query before executing it. If an error occurs, revise the query and try again.
        
        Avoid any DML statements (CREATE, INSERT, UPDATE, DELETE, DROP, etc.).
        
        If the question is unrelated to the database, return "I'm unable to provide an answer for that. This information is not available."
        
        Format your answers in Markdown, and if the format is a table, then make it a bordered table.
        
        Use your knowledge for questions related to the database when you do not have context.
        
        If the user does not pass any input, then return "I'm unable to provide an answer for that."
        
        Below are the column names with detailed descriptions that may assist in answering the user's query:
        
        """
        
        column_metadata = """
        - **SMC**: The SMC (Supplier Manufacturer Code) uniquely identifies the manufacturer or supplier of the items.
        - **Account_Code**: The unique code assigned to the account for the purpose of financial tracking or reporting.
        - **Account_Name**: The name associated with the account, typically representing the entity or individual that holds the account.
        - **Account_Details**: Additional details or descriptions about the account, including terms, conditions, and other relevant information.
        - **Analysis_Code**: The code used for categorizing or analyzing transactions or items, typically used for reporting or analysis.
        - **Analysis_Name**: The name associated with the analysis code, providing more context about the categorization or analysis purpose.
        - **Sub_account_Code**: The unique code for a sub-account, which is a subdivision of a main account, allowing more detailed tracking of financial transactions.
        - **Sub_Account_Name**: The name associated with the sub-account, describing its purpose or the entity it represents.
        - **alt_Account_Code**: An alternative account code used for tracking or categorization purposes, often used for cross-referencing.
        - **alt_Account_Name**: An alternative name for the account, typically used for different accounting systems or standards.
        - **alt_Analysis_Code**: An alternative analysis code used to categorize or group transactions for reporting or analysis purposes.
        - **alt_Analysis_Name**: An alternative name for the analysis code, providing a different reference to the categorization or analysis process.
        - **alt_Sub_account_Code**: An alternative code for a sub-account, used in cases where different systems or standards require separate coding.
        - **alt_Sub_Account_Name**: An alternative name for the sub-account, providing a reference for cross-system compatibility or reporting.
        - **VesselName**: The name of the vessel involved in the transaction, purchase, or order.
        - **Vessel_Objectid**: A unique identifier for the vessel, often used in databases to associate records to a specific vessel.
        - **Vendorid**: A unique identifier for the vendor or supplier, used for managing the vendor relationship and transactions.
        - **OwnerID**: The identifier for the owner of the vessel or equipment, used for tracking ownership.
        - **Primary_Manager_id**: The ID of the primary manager responsible for overseeing the vessel or equipment operations.
        - **Vessel_Id1**: An additional or alternate identifier for the vessel, possibly used for legacy systems or specific classifications.
        - **pocategory_id**: The unique identifier for the purchase order category, used to classify different types of purchase orders.
        - **Maker_id**: The unique identifier for the manufacturer of the item or equipment, used for tracking and managing products from specific makers.
        - **ITEM_CATEGORY_id**: The identifier for the category of the item, used to classify items into different categories for better management and reporting.
        - **IMONumber**: The International Maritime Organization (IMO) number is a unique identifier assigned to ships for maritime safety and legal purposes.
        - **OWNERNAME**: The name of the owner of the item, vessel, or equipment.
        - **pocategory**: The category of the purchase order, used to define the type or classification of the order (e.g., maintenance, procurement, etc.).
        - **PoNumber**: The unique identifier for the purchase order, used to track and reference the order in the procurement system.
        - **APPROVAL_FLAG**: A flag indicating whether the purchase order has been approved. Typically a boolean value (e.g., True or False).
        - **ApprovedDate**: The date on which the purchase order was approved, marking the official authorization of the order.
        - **POSENTDATE**: The date when the purchase order was entered into the system, indicating the creation or registration date.
        - **poitemcount**: The total number of items in the purchase order, summarizing the count of individual items listed in the order.
        - **Title**: The title of the item or transaction, often describing the nature of the purchase order or the item involved.
        - **VENDORCODE**: A unique code assigned to the vendor or supplier, used for vendor identification and classification.
        - **VendorName**: The name of the vendor or supplier providing the goods or services in the transaction.
        - **VENDORCOUNTRY**: The country where the vendor or supplier is located, important for logistics, legal, and reporting purposes.
        - **VENDOREMAIL**: The email address of the vendor, used for communications related to the transaction or order.
        - **VENDORPHONE**: The phone number of the vendor, used for contacting the supplier for queries or updates.
        - **VENDORAPPROVALSTATUS**: The approval status of the vendor, indicating whether the vendor is approved for transactions or is under review.
        - **BaseCurrency**: The primary currency used in the purchase order or financial transaction, defining the standard for pricing and value.
        - **BaseAmount**: The total amount in the base currency for the purchase order, excluding any adjustments, taxes, or fees.
        - **SchdDeliveryPort**: The scheduled delivery port for the goods or equipment, marking the intended arrival location for the shipment.
        - **REQ_NOS**: The required number of items or units specified in the purchase order, often used for inventory or fulfillment purposes.
        - **ENQNOS**: The associated number of the inquiry, possibly related to a request for quotation or inquiry process before purchase.
        - **GRNNO_AGENT_WAREHOUSE**: The Goods Receipt Note (GRN) number associated with the agent's warehouse, tracking goods receipt in the warehouse system.
        - **GRNNO_VESSEL**: The Goods Receipt Note (GRN) number related to the vessel, indicating the goods received on board the vessel.
        - **EQUIPMENTCODE**: A unique code assigned to a piece of equipment for identification and tracking.
        - **EQUIPMENTNAME**: The name of the equipment being referenced in the transaction or order.
        - **ParentCode**: The code for the parent item or category, often used to group items or transactions under a common parent classification.
        - **ParentName**: The name associated with the parent item or category, used to give more context to the classification of the item.
        - **Maker**: The manufacturer of the item, equipment, or vessel, responsible for the creation or production of the item.
        - **EQUIPMENT_TYPE**: The type or classification of the equipment, used for grouping or categorizing similar types of equipment.
        - **DrawingNo**: The number associated with the technical drawing or blueprint of the item or equipment, used for reference in design or manufacturing.
        - **SerialNo**: The unique serial number assigned to a specific piece of equipment, used for identification and tracking.
        - **MODEL**: The model name or number associated with the item, used for identifying the specific version or variant of the product.
        - **PART_NUMBER**: The unique identifier for a specific part of the equipment, used for inventory and replacement tracking.
        - **ITEM_ID**: A unique identifier for the item in the system, used to track the specific item in inventory or procurement.
        - **ITEM_DESCRIPTION**: A detailed description of the item, providing information about its features, specifications, and use cases.
        - **SERVICE_DESCRIPTION**: The description of the service provided with the item, detailing the nature of the service.
        - **DRAWING_NUMBER**: A reference number for a technical or engineering drawing related to the item, used for design and manufacturing.
        - **WEIGHT**: The weight of the item, typically used for shipping, inventory, and logistical purposes.
        - **PACKING_UOM**: The unit of measurement for the packing of the item, such as box, pallet, etc.
        - **UNIT_PRICE**: The price per unit of the item, used for calculating costs, pricing, and invoicing.
        - **QUANTITY**: The number of items or units in the purchase order or transaction.
        - **VENDOR_REMARKS**: Any additional remarks or comments from the vendor, typically related to terms, conditions, or special considerations.
        - **REMARKS_TO_VENDOR**: Notes or remarks addressed to the vendor, providing additional instructions or requests.
        - **ITEM_CATEGORY**: The category or classification of the item, often used for grouping items into different types or segments.
        - **ITEM_SECTION**: A specific section or subgroup within the item category, used for further classification and reporting.
        - **ITEM_CODE**: A unique code assigned to the item, used for identifying and tracking it in the system.
        - **UOM**: The unit of measurement for the item, such as kilogram, meter, etc.
        - **PO_USD_VALUE**: The total value of the purchase order in USD, used for financial tracking and reporting.
        - **po_amount_usd**: The amount of the purchase order in USD, often used for financial reconciliation.
        - **MD_REQUIRED**: Indicates whether a Material Data (MD) is required for the item.
        - **SDoC_REQUIRED**: Indicates whether a Supplier Declaration of Conformity (SDoC) is required for the item.
        - **UNIT_PRICE_USD**: The price per unit of the item in USD, used for international pricing or currency conversions.
        - **Received_Qty**: The quantity of items received, used for inventory tracking and logistics.
        - **Po_ApprovedDate**: The date when the purchase order was approved, used for tracking approval timelines.
        - **Po_Title**: The title or name associated with the purchase order, often used for categorization or easy reference.
        - **EQUIPMENT_ParentCode**: The parent code for the equipment, used for hierarchical tracking.
        - **EQUIPMENT_ParentName**: The parent name for the equipment, used for categorizing and tracking equipment in a group.
        - **ULTIMATE_OWNER_Name**: The name of the ultimate owner of the item or equipment, representing the highest level of ownership.
        - **PartNumber**: The part number associated with the item, used for identification and inventory purposes.
        - **UnitPrice**: The price per unit of the item, used for cost calculations and invoicing.
        - **ItemDescription**: A detailed description of the item, providing information about its specifications, use cases, or features.
        - **ReceivedQuantity**: The quantity of the items that have been received against the purchase order, often used in inventory and shipment tracking.
        """
        
        suffix = """
        If asked about the structure of the tables/database, respond with "I can answer questions from this DB but not the structure of this DB." 
        You are allowed to share the data inside the DB. Just don’t share information pertaining to the structure and column names. 
        
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

# Start the scheduler on app startup
@app.on_event("startup")
async def startup():
    scheduler.start()

# Shutdown the scheduler on app shutdown
@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()
