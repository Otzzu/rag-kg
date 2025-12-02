from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, PeftConfig
import torch

class TextToCypher:
    def __init__(self, schema: str):
        self._schema = schema
        # self._pipe = pipeline("text-generation", model="neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0", device_map="auto")
        # self._pipe = pipeline("text-generation", model="VoErik/cypher-gemma", device_map="auto")
        model_name = "neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self._instruction = (
            "Generate Cypher statement to query a graph database. "
            "Use only the provided relationship types and properties in the schema. \n"
            "Schema: {schema} \n Question: {question}  \n Cypher output: "
        )
    
    def prepare_chat_prompt(self, question, schema) -> list[dict]:
        chat = [
            {
                "role": "user",
                "content": self._instruction.format(
                    schema=schema, question=question
                ),
            }
        ]
        return chat
    
    def postprocess_output_cypher(self, output_cypher: str) -> str:
        # Remove any explanation. E.g.  MATCH...\n\n**Explanation:**\n\n -> MATCH...
        # Remove cypher indicator. E.g.```cypher\nMATCH...```` --> MATCH...
        # Note: Possible to have both:
        #   E.g. ```cypher\nMATCH...````\n\n**Explanation:**\n\n --> MATCH...
        partition_by = "**Explanation:**"
        output_cypher, _, _ = output_cypher.partition(partition_by)
        output_cypher = output_cypher.strip("`\n")
        output_cypher = output_cypher.lstrip("cypher\n")
        output_cypher = output_cypher.strip("`\n ")
        return output_cypher

    def __call__(self, question: str, chat_history: str = "", previous_error: str = ""):
        new_message = self.prepare_chat_prompt(question=question, schema=self._schema)
        prompt = self._tokenizer.apply_chat_template(new_message, add_generation_prompt=True, tokenize=False)
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True)

        # Any other parameters
        model_generate_parameters = {
            "top_p": 0.9,
            "temperature": 0.1,
            "max_new_tokens": 256,
            "do_sample": False,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        inputs.to(self._model.device)
        self._model.eval()
        with torch.no_grad():
            tokens = self._model.generate(**inputs, **model_generate_parameters)
            tokens = tokens[:, inputs.input_ids.shape[1] :]
            raw_outputs = self._tokenizer.batch_decode(tokens, skip_special_tokens=True)
            outputs = [self.postprocess_output_cypher(output) for output in raw_outputs]

        print("Raw generated text:", raw_outputs)
        return outputs

if __name__ == "__main__":
    with open("schema.txt") as fp:
        schema = fp.read().strip()

    print("Preparing pipeline ....")
    ttc = TextToCypher(schema)

    print("Generating ...")
    cypher = ttc("Which players currently have access to a 2:1 special port?")
    print(cypher)
