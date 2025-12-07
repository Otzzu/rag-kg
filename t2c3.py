from backend.text_to_cypher_v3 import TextToCypher
from backend.config import load_config

if __name__ == "__main__":
    with open("schema.txt") as fp:
        schema = fp.read().strip()
    cfg = load_config()

    print("Preparing pipeline ....")
    ttc = TextToCypher(schema=schema, config=cfg)

    print("Generating ...")
    cypher = ttc("Which players currently have access to a 2:1 special port?")
    print(cypher)
