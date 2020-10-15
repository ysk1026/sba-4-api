from com_sba_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey

class Item(Base):
    __tablename__ = "Items"
    
    id = Column(Integer, primary_key=True, index=True)    
