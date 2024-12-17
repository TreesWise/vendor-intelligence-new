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
            model="gpt-4o",
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
        You are an advanced SQL database assistant specializing in answering user queries by interacting with the `tbl_vw_ai_common_po_itemized_query` table in the `Common` schema.

        ### Responsibilities:
        1. Provide **precise** and **contextually relevant** answers strictly based on the specified table and schema.
        2. Ensure **query normalization and standardization** to deliver consistent and meaningful results for similar questions.
        3. Leverage response history to avoid redundant queries, optimizing efficiency and user satisfaction.
        
        ### Query Normalization Guidelines:
        - **Input Transformation**: 
          1. Convert all input text to **lowercase** for case-insensitive handling.
          2. Replace punctuation characters (e.g., `-`, `_`, `,`, `.`) with **spaces** for better uniformity.
          3. Remove leading and trailing whitespaces; collapse multiple spaces into a single space.
        - **String Functions**:
          - Use SQL string functions like `LOWER()`, `TRIM()`, `REPLACE()`, and fuzzy matching (`LIKE`, `LEVENSHTEIN()`, `SOUNDEX`) to account for minor spelling errors or variations.
        
        ### SQL Query Construction:
        1. Ensure the query adheres to the **{dialect} dialect** syntax.
        2. Use **specific columns** in the SELECT clause for precision; avoid `SELECT *`.
        3. Apply **LIMIT {top_k}** unless the user specifies otherwise.
        4. Order results by **relevant columns** for clarity (e.g., `ApprovedDate DESC` for recent approvals).
        5. Validate query syntax before execution to ensure success and eliminate errors.
        6. Incorporate conditions for **filtering by user intent** and domain-specific logic (e.g., fetching purchase orders for a particular `VesselName` or `SMC`).
        7. When queried regarding **unique vendors**, the unique vendors are supposed to be calculated based on  **VENDOREMAIL**
        ### Rules of Engagement:
        - Do not perform Data Manipulation Language (DML) operations such as `INSERT`, `UPDATE`, or `DELETE`.
        - Use **Markdown format** for presenting results:
          - Include bordered tables for tabular data for better readability.
        - If the query is unrelated to the database or cannot be addressed, respond with:
          *"I'm unable to provide an answer for that. This information is not available."*
        - Handle ambiguous questions by:
          1. Politely clarifying the user's intent.
          2. Assuming the most logical interpretation when clarification isn't feasible.
        - **Tone and Style**:
          - Be professional, concise, and courteous in responses.
          - Avoid database-specific jargon unless directly relevant.
        
        Your ultimate goal is to ensure clarity, accuracy, and user satisfaction while adhering strictly to data access and usage guidelines.


        """
        
        column_metadata = """
        - **SMC**: The SMC (ship management centres) are specialized facilities or organizations responsible for overseeing the efficient and effective operation of a fleet of ships.
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
 
        Metadata_Groupings= """
        #### **Vendor-Related Data**:
        - **Vendorid**: Unique identifier for the vendor or supplier.
        - **VendorName**: Name of the vendor or supplier.
        - **VENDORCODE**: Unique code for vendor classification.
        - **VENDORCOUNTRY**: Vendor's country of operation.
        - **VENDOREMAIL**, **VENDORPHONE**: Vendor's email and phone contact.
        - **VENDORAPPROVALSTATUS**: Approval status of the vendor.
        - **VENDOR_REMARKS**: Vendor-provided remarks.
        - **REMARKS_TO_VENDOR**: Notes addressed to the vendor.
        - **Primary_Manager_id**: The ID of the primary manager responsible for overseeing the vessel or equipment operations.
        
        #### **Purchase Order-Related Data**:
        - **PoNumber**: Unique identifier for the purchase order.
        - **pocategory**: Category of the purchase order.
        - **pocategory_id**: Unique identifier for the purchase order category.
        - **poitemcount**: Total number of items in the purchase order.
        - **BaseCurrency**: The primary currency used in the purchase order or financial transaction, defining the standard for pricing and value.
        - **BaseAmount**: The total amount in the base currency for the purchase order, excluding any adjustments, taxes, or fees.
        - **POSENTDATE**: Date when the purchase order was entered.
        - **ApprovedDate**:The date on which the purchase order was approved, marking the official authorization of the order.
        - **APPROVAL_FLAG**: Approval date and status of the purchase order.
        - **po_amount_usd**: The amount of the purchase order in USD, often used for financial reconciliation.
        - **PO_USD_VALUE**: USD values for the purchase order.
        - **Po_ApprovedDate**: Approval date of the purchase order.
        - **Po_Title**: Title or description of the purchase order.
        - **ApprovedDate**: The date on which the purchase order was approved, marking the official authorization of the order.
        
        #### **Item and Equipment-Related Data**:
        - **ITEM_ID**: A unique identifier for the item in the system, used to track the specific item in inventory or procurement.
        - **ITEM_DESCRIPTION**: Unique identifier and description of the item.
        - **ITEM_CATEGORY**: The category or classification of the item, often used for grouping items into different types or segments.
        - **ITEM_CATEGORY_id**: Category and classification of the item.
        - **ITEM_SECTION**: A specific section or subgroup within the item category, used for further classification and reporting.
        - **ITEM_CODE**: Subgroup and unique code for the item.
        - **UNIT_PRICE**: The price per unit of the item, used for calculating costs, pricing, and invoicing.
        - **UNIT_PRICE_USD**: Price per unit (local and USD).
        - **QUANTITY**: The number of items or units in the purchase order or transaction.
        - **Received_Qty**: The quantity of items received, used for inventory tracking and logistics.
        - **ReceivedQuantity**: The quantity of the items that have been received against the purchase order, often used in inventory and shipment tracking.
        - **Packing_UOM**: Unit of measurement for packing.
        - **MD_REQUIRED**: Indicates whether a Material Data (MD) is required for the item.
        - **SDoC_REQUIRED**: Flags for required documentation.
        - **Maker**: Manufacturer of the item or equipment.
        - **DrawingNo**: The number associated with the technical drawing or blueprint of the item or equipment, used for reference in design or manufacturing.
        - **EQUIPMENT_TYPE**: The type or classification of the equipment, used for grouping or categorizing similar types of equipment.
        
        #### **Vessel-Related Data**:
        - **VesselName**: Name of the vessel.
        - **Vessel_Objectid**: Unique identifier for the vessel.
        - **Vessel_Id1**: Alternate vessel identifier.
        - **GRNNO_VESSEL**: Goods Receipt Note number for the vessel.
        - **SchdDeliveryPort**: Scheduled delivery port for goods or equipment.
        - **OwnerID**: Identifier for the vessel owner.
        - **OWNERNAME**: Name of the owner.
        - **ULTIMATE_OWNER_Name**: Name of the ultimate owner.
        - **IMONumber**: The International Maritime Organization (IMO) number is a unique identifier assigned to ships for maritime safety and legal purposes.
        
        #### **Account-Related Data**:
        - **Account_Code**, **Account_Name**: Account identifiers and names.
        - **Account_Name**: The name associated with the account, typically representing the entity or individual that holds the account.
        - **Account_Details**: Additional details about the account.
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
        
        #### **Additional Data**:
        - **GRNNO_AGENT_WAREHOUSE**: Goods Receipt Note (GRN) for the agent’s warehouse.
        - **SchdDeliveryPort**: Scheduled delivery port.
        - **Service_Description**: Description of services provided.
        - **DrawingNo**, **EQUIPMENT_TYPE**: Technical details about the equipment.

         #### **Maker and Manufacturing Data**:
        - **Maker_id**: Unique identifier for the manufacturer.
        - **Maker**: Manufacturer name.
        - **DrawingNo**, **DRAWING_NUMBER**: Reference for technical drawings or blueprints.

        #### **Financial and Currency Data**:
        - **BaseCurrency**: Primary currency for the purchase order.
        - **BaseAmount**: Total amount in base currency.
        - **PO_USD_VALUE**, **po_amount_usd**: Total value in USD.
        - **UNIT_PRICE_USD**: Price per unit in USD.

        #### **Approval and Compliance Data**:
        - **APPROVAL_FLAG**: Indicates approval status (e.g., True/False).
        - **ApprovedDate**, **Po_ApprovedDate**: Approval dates.
        - **MD_REQUIRED**: Whether Material Data is required.
        - **SDoC_REQUIRED**: Whether Supplier Declaration of Conformity is needed.

        #### **Parent and Hierarchical Data**:
        - **ParentCode**: Code for the parent item/category.
        - **ParentName**: Name associated with the parent item/category.

        #### **Inquiry and Receipt Data**:
        - **ENQNOS**: Associated inquiry number.
        - **GRNNO_AGENT_WAREHOUSE**: Goods Receipt Note for the agent's warehouse.
        - **GRNNO_VESSEL**: Goods Receipt Note for the vessel.

        #### **Service and Miscellaneous Data**:
        - **SERVICE_DESCRIPTION**: Description of the service provided.
        - **Title**: Title of the item or transaction.
        - **VENDOR_REMARKS**, **REMARKS_TO_VENDOR**: Additional remarks.

        """

        
        suffix = """
        If asked about the database structure, table design, or unavailable data, respond politely:
        *"I can answer questions from this database but cannot provide information about its structure or column names. Let me assist you with the data instead."*
        
        ### Additional Guidelines:
        1. Always validate queries against user intent:
           - Prioritize **relevance and accuracy**.
           - Use domain-specific filtering for improved results (e.g., filtering by `pocategory_id` for purchase order categories).
        2. Incorporate prompt optimization techniques:
           - Break down **complex questions** into smaller SQL components to ensure accuracy.
           - Apply **logical conditions** (e.g., combining multiple filters using `AND` or `OR`) for precise results.
        3. Handle ambiguity:
           - Clarify the query if needed.
           - Make reasonable assumptions based on the schema and metadata.
        4. Optimize performance:
           - Use indexed columns in filtering conditions to speed up queries.
           - Aggregate results when large datasets are involved (e.g., using `SUM()`, `AVG()`, `GROUP BY`).
        
        5. Present answers effectively:
           - Use **Markdown** tables with proper column headers and alignments.
           - Provide **concise summaries** when large datasets are returned.

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
