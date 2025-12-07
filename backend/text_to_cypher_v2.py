from openai import OpenAI
from backend.config import Config

EXAMPLES = """
    Question: List all players and their total victory points.
    Cypher: MATCH (p:Player)-[:HAS_PIECE]->(piece:Piece)
            RETURN p.name AS playerName, SUM(piece.vp) AS totalVP
            ORDER BY totalVP DESC;

    Question: Find all settlements owned by the player whose name contains 'mesach'.
    Cypher: MATCH (p:Player)-[:OWNS]->(s:Settlement)
            WHERE toLower(p.name) CONTAINS "mesach"
            RETURN s;

    Question: Get all tiles that produce lumber and their dice numbers.
    Cypher: MATCH (t:Tile)
            WHERE toLower(t.resource) CONTAINS "lumber"
            RETURN t.name AS tileName, t.diceNumber AS diceNumber;

    Question: List all roads built by players whose name contains 'ivan'.
    Cypher: MATCH (p:Player)-[:BUILDS]->(r:Road)
            WHERE toLower(p.name) CONTAINS "ivan"
            RETURN p.name AS playerName, r;

    Question: Find all harbor nodes and the players connected to them.
    Cypher: MATCH (h:Harbor)<-[:ADJACENT_TO]-(i:Intersection)<-[:OWNS]-(p:Player)
            RETURN h, p;
"""

class TextToCypher:
    def __init__(
        self,
        schema: str,
        config: Config,
        model: str = "qwen/qwen3-coder:free",
    ):
        self._schema = schema
        self._model = model
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.get_openai_key(),
        )

        self._instruction = (
            "You are an expert Neo4j Cypher generator.\n"
            "Your task is to convert the user's natural language request into a single valid Cypher query.\n\n"
            "Rules:\n"
            "1. Use ONLY the labels, relationship types, and properties provided in the Schema.\n"
            "2. Do NOT invent new labels, relationships, or properties.\n"
            "3. Do NOT add any explanation or conversational text.\n"
            "4. Return ONLY the Cypher query (no markdown fences, no comments).\n"
            "5. For string matching filters, ALWAYS use `toLower()` and `CONTAINS` to be case-insensitive.\n"
            "   Example: use `toLower(l.name) CONTAINS \"forest\"` instead of `l.name = \"Forest\"`.\n\n"
            "Here are a few examples of correct behavior:\n"
            f"{EXAMPLES.strip()}\n\n"
            "Now, using ONLY the schema below, answer the question with a single Cypher query.\n\n"
            "Schema:\n{schema}\n\n"
            "Question: {question}\n"
            "Cypher:"
        )

    def prepare_chat_prompt(
        self,
        question: str,
        schema: str,
    ) -> list[dict]:

        content = self._instruction.format(schema=schema, question=question)
        chat = [
            {
                "role": "system",
                "content": (
                    "You are a strict Cypher query generator. "
                    "You must respond with ONLY a single valid Cypher query. "
                    "Do not add explanations, comments, or markdown fences."
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        return chat

    def postprocess_output_cypher(self, output_cypher: str) -> str:
        partition_by_list = ["**Explanation:**", "Explanation:", "EXPLANATION:"]
        for p in partition_by_list:
            if p in output_cypher:
                output_cypher, _, _ = output_cypher.partition(p)
                break

        output_cypher = output_cypher.replace("```cypher", "")
        output_cypher = output_cypher.replace("```", "")

        if "Cypher:" in output_cypher:
            output_cypher = output_cypher.split("Cypher:", 1)[-1]

        output_cypher = output_cypher.strip("`\n ")
        output_cypher = output_cypher.replace("\\n", " ")
        return output_cypher.strip()

    def __call__(
        self,
        question: str,
    ):
        messages = self.prepare_chat_prompt(
            question=question,
            schema=self._schema,
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.1,
        )

        raw_output = response.choices[0].message.content or ""
        processed_output = self.postprocess_output_cypher(raw_output)

        print("Raw generated text:", raw_output)
        print("Post-processed Cypher:", processed_output)

        return [processed_output]


