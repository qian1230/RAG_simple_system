"""数据库配置管理

统一管理Qdrant和Neo4j的配置，支持从环境变量读取配置。
"""

import os
from typing import Dict, Any, Optional


class DatabaseConfig:
    """数据库配置类"""
    
    def __init__(self):
        # Qdrant配置
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # Neo4j配置
        self.neo4j_uri = os.getenv("NEO4J_URI_BOLT")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """获取Qdrant配置"""
        config = {}
        if self.qdrant_url:
            config["url"] = self.qdrant_url
        if self.qdrant_api_key:
            config["api_key"] = self.qdrant_api_key
        return config
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """获取Neo4j配置"""
        config = {}
        if self.neo4j_uri:
            config["uri"] = self.neo4j_uri
        if self.neo4j_username:
            config["username"] = self.neo4j_username
        if self.neo4j_password:
            config["password"] = self.neo4j_password
        if self.neo4j_database:
            config["database"] = self.neo4j_database
        return config
    
    def is_qdrant_configured(self) -> bool:
        """检查Qdrant是否配置"""
        return bool(self.qdrant_url and self.qdrant_api_key)
    
    def is_neo4j_configured(self) -> bool:
        """检查Neo4j是否配置"""
        return bool(self.neo4j_uri and self.neo4j_username and self.neo4j_password)


def get_database_config() -> DatabaseConfig:
    """获取数据库配置实例"""
    return DatabaseConfig()
