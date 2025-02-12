# from typing import Union, List
# from langchain_core.pydantic_v1 import BaseModel, Field
# import pydantic as pd

# class ModelInput(pd.BaseModel):
#     Db_Name: str = str
#     User_Query: str = str

from typing import Optional,List
from pydantic import BaseModel, Field

class ModelInput(BaseModel):
    db_name: str = Field("vendor intelligence", description="The name of the database")  # Required field
    user_query: Optional[str] = Field(None, description="The query provided by the user")  # Optional field
    item_id: Optional[List[int]] = []  # Allow multiple item IDs
    port_id: Optional[List[int]] = []  # Allow multiple port IDs
    item_name: Optional[List[str]] = []  # Allow multiple item names
    port_name: Optional[List[str]] = []  # Allow multiple port names
