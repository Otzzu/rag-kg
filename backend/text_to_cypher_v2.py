from transformers import pipeline

EXAMPLES = """
Question: 
Cypher: 

Question: 
Cypher: 

Question: 
Cypher:
"""

PROMPT_TEMPLATE = """
<INSTRUCTION>
You are an expert Neo4j Cypher generator.
Your task is to convert the user's natural language request into a single valid Cypher query.

Rules:
1. Use ONLY the labels, relationship types, and properties provided in the Schema.
2. Do NOT invent new labels, relationships, or properties.
3. Do NOT add any explanation or conversational text.
4. Return ONLY the Cypher query.
5. For string matching filters, ALWAYS use 'toLower()' and 'CONTAINS' to be case-insensitive. Example: Instead of "l.name = 'Forest'", use "toLower(l.name) CONTAINS 'forest'".
</INSTRUCTION>

<SCHEMA>
[SCHEMA]
</SCHEMA>

<EXAMPLES>
[EXAMPLES]
</EXAMPLES>

<TASK>
Question: [QUESTION]
Cypher:"""

class TextToCypher:
    def __init__(self, schema: str):
        self._schema = schema
        self._pipe = pipeline("text-generation", model="VoErik/cypher-gemma")

    def __call__(self, question: str):
        prompt = PROMPT_TEMPLATE
        prompt = prompt.replace("[SCHEMA]", self._schema)
        prompt = prompt.replace("[EXAMPLES]", EXAMPLES.strip())
        prompt = prompt.replace("[QUESTION]", question)

        output = self._pipe(
            prompt,
            max_new_tokens=256,
            return_full_text=False,
            temperature=0.1
        )[0]

        generated_text = output["generated_text"]
        generated_text = generated_text.replace("```cypher", "").replace("```", "")

        if "Cypher:" in generated_text:
            generated_text = generated_text.split("Cypher:")[-1]

        return generated_text.strip()