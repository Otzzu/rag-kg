from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = """
    You are a helpful and strict assistant for Catan Base Game Rules & Strategy.
    Your sole purpose is to answer the user's question using ONLY the provided database evidence.
    
    RULES:
    1. Do NOT use outside knowledge. Do NOT hallucinate.
    2. Keep answers concise and direct.
""".strip()

USER_PROMPT_TEMPLATE = """
    ### Schema
    [SCHEMA]

    ### User Question
    [QUESTION]
    
    ## Query
    [QUERY]

    ### Database Evidence
    [QUERY_RESULT_STR]
    ([EVIDENCE_STATUS])

    Based on the evidence above, provide the answer:
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
            return "I couldn't find any information about that in the database."
        
        if "(error occurred)" in query_result_str:
            return "I'm sorry, there was an error retrieving the data."

        evidence_status = "Data found."

        user_content = USER_PROMPT_TEMPLATE
        user_content = user_content.replace("[SCHEMA]", self._schema)
        user_content = user_content.replace("[QUESTION]", question)
        user_content = user_content.replace("[QUERY]", query)
        user_content = user_content.replace("[QUERY_RESULT_STR]", query_result_str)
        user_content = user_content.replace("[EVIDENCE_STATUS]", evidence_status)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
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