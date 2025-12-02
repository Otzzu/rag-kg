from backend.text_to_cypher_v3 import TextToCypher

if __name__ == "__main__":
    with open("schema.txt") as fp:
        schema = fp.read().strip()

    print("Preparing pipeline ....")
    ttc = TextToCypher(schema)

    print("Generating ...")
    cypher = ttc("Which players currently have access to a 2:1 special port?")
    print(cypher)
