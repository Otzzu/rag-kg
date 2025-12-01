import streamlit as st
import os
from backend.database import GraphDatabaseDriver
from backend.text_to_cypher_v2 import TextToCypher
from backend.response_generator_v2 import ResponseGenerator
from backend.config import load_config


st.set_page_config(page_title="Tugas Proyek II - RAG", layout="centered")
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
    
    return TextToCypher(schema), ResponseGenerator(schema), load_config("config.toml") 

with st.spinner("Loading system..."):
    ttc, generator, config = init_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_chat_history_str():
    history_str = ""
    recent_msgs = st.session_state.messages[-3:] 
    for msg in recent_msgs:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    return history_str

if question := st.chat_input("Masukkan pertanyaan..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_container = st.status("Thinking...", expanded=True)
        
        last_error = ""
        final_context_str = ""
        final_cypher = ""
        success = False
        
        chat_history = get_chat_history_str()

        for attempt in range(config.get("max_retries", 3)):
            with status_container:
                if attempt > 0:
                    st.write(f"Percobaan ke-{attempt+1}: Memperbaiki query...")
                else:
                    st.write("Menerjemahkan pertanyaan ke Cypher...")
                
                # 1. Generate Query
                cypher_query = ttc(question, chat_history, last_error)
                
                st.code(cypher_query, language="cypher")
                final_cypher = cypher_query
                
                # 2. Coba Eksekusi di Database
                try:
                    with GraphDatabaseDriver(config) as driver:
                        results = driver.execute_query(cypher_query)
                    
                    if results:
                        st.write(f"Ditemukan {len(results)} data.")
                        final_context_str = "\n".join([str(x) for x in results])
                    else:
                        st.write("Query valid tapi data kosong.")
                        final_context_str = "(no result)"
                    
                    success = True
                    break
                    
                except Exception as e:
                    last_error = str(e)
                    st.error(f"Error Database: {last_error}")
        
        if not success:
            final_context_str = f"(error occurred) Detail: {last_error}"
            st.warning("Gagal mengambil data. Mengirim laporan error ke asisten...")

        # 3. Generate Jawaban Akhir
        status_container.update(label="Menyusun Jawaban...", state="running")
        
        final_answer = generator(question, final_cypher, final_context_str)
        
        status_container.update(label="Selesai!", state="complete", expanded=False)
        
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})