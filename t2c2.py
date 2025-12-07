from backend.config import load_config
from backend.text_to_cypher_v2 import TextToCypher

if __name__ == "__main__":
    with open("schema.txt", encoding="utf-8") as fp:
        schema = fp.read().strip()

    print("Preparing pipeline ....")
    config = load_config()
    ttc = TextToCypher(schema=schema, config=config, model="kwaipilot/kat-coder-pro:free")
    
    print("Generating ...")
    cypher_list = ttc("Which players currently have access to a 2:1 special port?")
    print(cypher_list)
