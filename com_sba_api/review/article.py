from com_sba_api.ext.db import Base
from sqlalchemy import Column, Integer, String, ForeignKey, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, LONGTEXT

class Article(Base):
    __tablename__ = "articles"
    __table_args__ = {'mysql_collate' : 'utf8_general_ci'}
    
    id = Column(Integer, primary_key=True, index=True)
    user = Column("user.id")
    item = Column("item.id")
    title = Column(VARCHAR(30))
    content = Column(VARCHAR(30))
    
engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8', encoding='utf8', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind = engine)
session = Session()
# session.add(Article(user = user, item = item, title = ))
session.commit()