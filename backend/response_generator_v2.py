from transformers import AutoModelForCausalLM, AutoTokenizer

DOMAIN_NAME = "Catan Base Game Rules & Strategy"
ROLE_DESCRIPTION = f"""
    You are a smart and friendly assistant specialized in analyzing {DOMAIN_NAME} data.
    Your task is to answer user questions based strictly on the data found in the database.
    You should use a helpful, professional, yet conversational tone.
"""

PROMPT_TEMPLATE = """
    ### System
    [ROLE_DESCRIPTION]

    IMPORTANT RULES:
    1. Answer ONLY based on the information provided in the Evidence section.
    2. If the Evidence indicates NO data, apologize politely and state that you could not find that information in the database.
    3. Do NOT hallucinate or invent facts not present in the evidence.
    4. If the data is technical, explain it in simple terms.

    ### Context
    Domain: [DOMAIN_NAME]
    Schema:
    [SCHEMA]

    ### User Question
    [QUESTION]

    ### Generated Cypher Query
    [QUERY]

    ### Evidence
    [QUERY_RESULT_STR]
    ([EVIDENCE_STATUS])

    ### Answer
""".strip()

class ResponseGenerator:
    def __init__(self, schema: str):
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._schema = schema

    def __call__(self, question: str, query: str, query_result_str: str):
        if query_result_str.strip() in ["(no result)", "[]", ""]:
            evidence_status = "NO data found in the database."
        elif "(error occurred)" in query_result_str:
            evidence_status = "An ERROR occurred while fetching data."
        else:
            evidence_status = "Valid data found above."

        prompt = PROMPT_TEMPLATE
        prompt = prompt.replace("[ROLE_DESCRIPTION]", ROLE_DESCRIPTION)
        prompt = prompt.replace("[DOMAIN_NAME]", DOMAIN_NAME)
        prompt = prompt.replace("[SCHEMA]", self._schema)
        prompt = prompt.replace("[QUESTION]", question)
        prompt = prompt.replace("[QUERY]", query)
        prompt = prompt.replace("[QUERY_RESULT_STR]", query_result_str)
        prompt = prompt.replace("[EVIDENCE_STATUS]", evidence_status)

        messages = [
            {"role": "system", "content": f"You are a helpful assistant for {DOMAIN_NAME}."},
            {"role": "user", "content": prompt}
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = (
            self._tokenizer([text], return_tensors="pt")
            .to(self._model.device)
        )
        
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.7 
        )
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids,
                                             generated_ids)
        ]
        
        response = self._tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        if "###" in response:
            response = response.split("###")[0]

        return response.strip()