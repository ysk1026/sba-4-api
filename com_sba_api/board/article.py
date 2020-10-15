from com_sba_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey

class Article(Base):
    __tablename__ = "articles"
    
    id = Column(Integer, primary_key=True, index=True)