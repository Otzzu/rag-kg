import tomllib
import os

class Config:
    def __init__(self, data: dict[str]):
        self._data = data
    
    def get_neo4j_driver_kwargs(self):
        neo4j_data = self._data["neo4j"]
        return {
            "uri": neo4j_data["database_uri"],
            "auth": (neo4j_data["username"], neo4j_data["password"])
        }
    
    def get_neo4j_database_name(self):
        neo4j_data = self._data["neo4j"]
        return neo4j_data["database_name"]
    
    def get_openai_key(self):
        openai_data = self._data["openai"]
        return openai_data["openai_api_key"]

def load_config(toml_path: str = "config.toml"):
    if not os.path.exists(toml_path):
        toml_path = os.path.join(os.path.dirname(__file__), "..", toml_path)
    with open(toml_path, mode="rb") as fp:
        return Config(tomllib.load(fp))
