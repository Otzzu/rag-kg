from backend.config import load_config
from backend.text_to_cypher_v2 import TextToCypher


if __name__ == "__main__":

    with open("schema.txt", encoding="utf-8") as fp:
        schema = fp.read().strip()

    print("Preparing pipeline ....")
    config = load_config()
    # ttc = TextToCypher(schema=schema, config=config, model="qwen/qwen3-235b-a22b:free")
    # ttc = TextToCypher(schema=schema, config=config, model="nousresearch/hermes-3-llama-3.1-405b:free")
    # ttc = TextToCypher(schema=schema, config=config, model="x-ai/grok-4.1-fast:free")
    # ttc = TextToCypher(schema=schema, config=config, model="tngtech/deepseek-r1t2-chimera:free")
    ttc = TextToCypher(schema=schema, config=config, model="kwaipilot/kat-coder-pro:free")
    

    print("Generating ...")
    cypher_list = ttc("Which players currently have access to a 2:1 special port?")
    print(cypher_list)
