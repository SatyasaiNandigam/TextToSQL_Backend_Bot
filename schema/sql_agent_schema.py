from pydantic import BaseModel

class SQLSchema(BaseModel):
    sql : str