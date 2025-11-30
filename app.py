import streamlit as st
import os
from backend.database import GraphDatabaseDriver
from backend.text_to_cypher_v2 import TextToCypher
from backend.response_generator_v2 import ResponseGenerator

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
    
    return TextToCypher(schema), ResponseGenerator(schema)

with st.spinner("Loading system..."):
    ttc, generator = init_resources()

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
            cypher_query = ttc(question)
            st.code(cypher_query, language="cypher")
            
            # 2. Execute Query
            st.write("Executing database query...")
            context_str = ""
            try:
                with GraphDatabaseDriver() as driver:
                    results = driver.execute_query(cypher_query)
                
                if results:
                    st.success(f"Found {len(results)} records.")
                    st.json(results, expanded=False)
                    context_str = "\n".join([str(x) for x in results])
                else:
                    st.warning("No data found.")
                    context_str = "(no result)"
            
            except Exception as e:
                st.error(f"Database error: {e}")
                context_str = "(error occurred)"
            # context_str = "36" ini buat tes saja
                
            # 3. Generate Answer
            st.write("Generating final response...")
            final_answer = generator(question, cypher_query, context_str)
            
            status.update(label="Done", state="complete", expanded=False)
        
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})