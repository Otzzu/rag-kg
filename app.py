import streamlit as st
import os
from backend.database import GraphDatabaseDriver
from backend.text_to_cypher_v2 import TextToCypher
from backend.response_generator_v2 import ResponseGenerator
from backend.config import load_config

st.set_page_config(
    page_title="Tugas Proyek II - RAG",
    layout="centered"
)

st.title("Knowledge Graph RAG")
st.caption("IF4070 Representasi Pengetahuan & Penalaran")

@st.cache_resource
def init_resources():
    schema_path = "schema.txt"
    if not os.path.exists(schema_path):
        st.error(f"File skema '{schema_path}' tidak ditemukan!")
        st.stop()
        
    with open(schema_path) as fp:
        schema = fp.read().strip()
    
    config = load_config()
    
    # return TextToCypher(schema, config, "kwaipilot/kat-coder-pro:free"), ResponseGenerator(schema), config
    return TextToCypher(schema, config), ResponseGenerator(schema), config

with st.spinner("Loading system..."):
    ttc, generator, config = init_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Masukkan pertanyaan..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Processing...", expanded=True) as status:
            
            # 1. Generate Cypher
            st.write("Generating Cypher query...")
            cypher_queries = ttc(question)

            if isinstance(cypher_queries, str):
                cypher_queries = [cypher_queries]

            if not cypher_queries:
                st.error("The model did not generate any Cypher query.")
                status.update(label="Failed", state="error", expanded=False)
                st.stop()

            for i, q in enumerate(cypher_queries, start=1):
                st.markdown(f"**Cypher Query {i}:**")
                st.code(q, language="cypher")

            # 2. Execute Query
            st.write("Executing database query...")
            all_results = []
            context_str_parts = []

            try:
                with GraphDatabaseDriver(config) as driver:
                    for i, q in enumerate(cypher_queries, start=1):
                        st.write(f"Running query {i}...")
                        res = driver.execute_query(q)

                        res = res or []
                        all_results.extend(res)

                        for row in res:
                            context_str_parts.append(str(row))

                if all_results:
                    st.success(f"Found {len(all_results)} total records from {len(cypher_queries)} query(ies).")
                    st.json(all_results, expanded=False)
                    context_str = "\n".join(context_str_parts)
                else:
                    st.warning("No data found from any query.")
                    context_str = "(no result)"

            except Exception as e:
                st.error(f"Database error: {e}")
                context_str = f"(error occurred: {e})"

            # 3. Generate Answer
            st.write("Generating final response...")

            combined_cypher = "\n\n".join(cypher_queries)
            final_answer = generator(question, combined_cypher, context_str)
            
            status.update(label="Done", state="complete", expanded=False)

        
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})