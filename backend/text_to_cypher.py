from transformers import pipeline

class TextToCypher:
    def __init__(self, schema: str):
        self._schema = schema
        # self._pipe = pipeline("text-generation", model="neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0", device_map="auto")
        self._pipe = pipeline("text-generation", model="VoErik/cypher-gemma", device_map="auto")

    def __call__(self, question: str, chat_history: str = "", previous_error: str = ""):
        output = self._pipe(
            [{
                "role": "user",
                "content": f"Question: {question} \n Schema: {self._schema}"}
            ],
            max_new_tokens=256,
            return_full_text=False
        )[0]
        generated_text = output["generated_text"]
        print("Raw generated text:", generated_text)
        generated_text = generated_text.replace(r"\n", "\n")
        return generated_text

if __name__ == "__main__":
    with open("schema.txt") as fp:
        schema = fp.read().strip()

    print("Preparing pipeline ....")
    ttc = TextToCypher(schema)

    print("Generating ...")
    cypher = ttc("Find all players that submit a comment \"GG!\".")
    print(cypher)
