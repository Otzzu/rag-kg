# Catan RAG - Knowledge Graph Question Answering System

This repository implements a Retrieval-Augmented Generation (RAG) system for answering questions about the Catan board game using a Neo4j knowledge graph. The system converts natural language questions into Cypher queries, executes them against the graph database, and generates natural language responses.

**Course:** IF4070 Knowledge Representation and Reasoning, STEI-ITB  

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Testing the System](#testing-the-system)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This RAG system bridges natural language and structured graph data by:
1. **Converting** user questions into Cypher queries (Text-to-Cypher)
2. **Executing** queries against a Neo4j knowledge graph containing Catan game data
3. **Generating** natural language responses from the query results

The system supports both **local models** and **cloud-based models** (via OpenRouter API) for different components.

---

## 🏗️ System Architecture

The system consists of three main components working in a pipeline:

```
User Question → Text-to-Cypher → Neo4j Database → Response Generator → Final Answer
```

### Component Details

| Component | Purpose | Implementation Options |
|-----------|---------|----------------------|
| **Text-to-Cypher** | Converts natural language to Cypher queries | • **Local**: Using local LLM models<br>• **OpenRouter**: Using cloud-based models (Qwen3-Coder) |
| **Database Driver** | Executes Cypher queries against Neo4j | Neo4j Python driver |
| **Response Generator** | Generates natural language answers from query results | • **Local**: Qwen2.5-0.5B-Instruct (runs on CPU)<br>• **V2**: Enhanced version with better prompting |

---

## 📁 Project Structure

```
rag-kg/
├── backend/                          # Core backend modules
│   ├── config.py                     # Configuration loader (reads config.toml)
│   ├── database.py                   # Neo4j database driver wrapper
│   ├── text_to_cypher.py            # Text-to-Cypher (basic version)
│   ├── text_to_cypher_v2.py         # Text-to-Cypher (improved, using OpenRouter)
│   ├── text_to_cypher_v3.py         # Text-to-Cypher (alternative version)
│   ├── response_generator.py        # Response generator (basic version)
│   └── response_generator_v2.py     # Response generator (improved prompts)
│
├── app.py                            # Streamlit web interface (main version)
├── rag.py                            # Simple CLI for RAG testing
├── schema.txt                        # Neo4j graph schema (node types, relationships)
├── knowledge-graph-2025-12-03T06-30-05.dump  # Neo4j database dump file
│
├── config_template.toml              # Template for configuration file
├── config.toml                       # Your actual configuration (gitignored)
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project metadata (uv/pip)
└── README.md                         # This file
```

### Key Files Explained

- **`app.py`**: Streamlit web application providing an interactive chat interface
- **`backend/text_to_cypher_v2.py`**: Uses OpenRouter API with Qwen3-Coder to generate Cypher queries
- **`backend/response_generator_v2.py`**: Uses local Qwen2.5-0.5B-Instruct model to generate answers
- **`backend/database.py`**: Handles Neo4j connection and query execution
- **`schema.txt`**: Documents the complete graph schema (nodes, relationships, properties)
- **`.dump` file**: Contains the actual Catan knowledge graph data

---

## ⚙️ How It Works

### 1️⃣ Text-to-Cypher Pipeline

When you ask a question like *"How many players are there?"*:

1. **Input Processing**: The question is combined with the graph schema from `schema.txt`
2. **LLM Generation**: 
   - **OpenRouter-based** (`text_to_cypher_v2.py`): Sends the question + schema to the Qwen3-Coder model via OpenRouter API
   - **Local** (alternative versions): Uses locally hosted models
3. **Post-processing**: Cleans up the generated Cypher query (removes markdown fences, explanations)
4. **Output**: Returns valid Cypher query(ies)

**Example:**
```
Question: "How many players are there?"
Generated Cypher: MATCH (p:Player) RETURN COUNT(p) AS playerCount
```

### 2️⃣ Database Execution

1. The generated Cypher query is sent to Neo4j via the `GraphDatabaseDriver`
2. Neo4j executes the query against the knowledge graph
3. Results are returned as structured data (dictionaries/lists)

**Example Result:**
```json
[{"playerCount": 3}]
```

### 3️⃣ Response Generation

1. **Input**: Original question + Cypher query + query results
2. **Local LLM**: Qwen2.5-0.5B-Instruct processes this context
3. **Safeguards**: Built-in checks prevent hallucination on empty results
4. **Output**: Natural language answer

**Example:**
```
"There are 3 players in the game."
```

### 🔄 Full Example Flow

```
User: "List all players and their victory points"
  ↓
Text-to-Cypher: MATCH (p:Player) RETURN p.name, p.vp ORDER BY p.vp DESC
  ↓
Neo4j Database: [{"p.name": "Filbert", "p.vp": 8}, {"p.name": "Ivan", "p.vp": 6}, ...]
  ↓
Response Generator: "Here are the players and their victory points: Filbert has 8 VP, Ivan has 6 VP..."
  ↓
User sees final answer
```

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11+** (recommended)
- **Neo4j Database** (AuraDB free tier or local instance)
- **OpenRouter API Key** (for cloud-based Text-to-Cypher)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd rag-kg
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
py -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you need GPU support for PyTorch, install the appropriate version:
```bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Set Up Neo4j Database

#### Option A: Using Neo4j AuraDB (Recommended)

1. Go to [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. **Save your credentials** (username, password, connection URI)

#### Option B: Using Local Neo4j

1. Download and install [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database
3. Start the database instance

### Step 5: Load the Knowledge Graph Dump

#### Using Neo4j Desktop/Browser:

1. Open Neo4j Browser
2. Stop your database instance
3. Use Neo4j Admin tools to restore the dump:

```bash
# Navigate to your Neo4j installation
neo4j-admin database load --from-path=<path-to-dump-file> neo4j
```

#### Using AuraDB:

1. Upload the dump file through the AuraDB console
2. Or manually recreate the graph structure using the schema in `schema.txt`

> **Important:** The dump file `knowledge-graph-2025-12-03T06-30-05.dump` contains the complete Catan game state including players, pieces, resources, and board configuration.

### Step 6: Configure the Application

1. **Copy the template configuration:**

```bash
cp config_template.toml config.toml
```

2. **Edit `config.toml` with your credentials:**

```toml
[neo4j]
database_uri = "neo4j+s://xxxxx.databases.neo4j.io"  # Your Neo4j connection URI
database_name = "neo4j"                               # Usually "neo4j" for AuraDB
username = "neo4j"                                    # Your username
password = "your-password-here"                       # Your password

[openai]
openai_api_key = "sk-or-v1-xxxxx"                    # Your OpenRouter API key
```

**Getting an OpenRouter API Key:**
1. Go to [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to "Keys" section
4. Create a new API key
5. Copy it to your `config.toml`

### Step 7: Verify Setup (Optional)

Test database connection:

```bash
python backend/database.py
```

This should connect to Neo4j and run a sample query.

---

## ▶️ Running the Application

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Chat-style interface
- Shows generated Cypher queries
- Displays query results (expandable JSON)
- Streams the final answer
- Conversation history

### CLI Testing

For quick testing without the web interface:

```bash
python rag.py
```

Enter your question when prompted.

---

## 🧪 Testing the System

### Sample Questions to Try

#### Basic Queries:
- "How many players are there?"
- "List all players"
- "What resources exist in the game?"

#### Intermediate Queries:
- "Show me all settlements owned by Filbert"
- "Which tiles produce lumber?"
- "List all players and their victory points"

#### Advanced Queries:
- "Which player has the most victory points?"
- "Find all hexes with dice number 6"
- "Show me all roads built by players whose name contains 'ivan'"
- "Which intersections are adjacent to harbors?"

### Expected Behavior

1. **Question Input**: Type your question in the chat
2. **Cypher Generation**: See the generated Cypher query (shown in code block)
3. **Query Execution**: View database results (JSON format)
4. **Answer Generation**: Read the natural language response

### Debugging Tips

If the system produces errors:

1. **Check the generated Cypher**: Look for syntax errors
2. **Verify database connection**: Ensure Neo4j is running and credentials are correct
3. **Review schema alignment**: Make sure queries use labels/properties from `schema.txt`
4. **Check API quota**: Verify your OpenRouter API key has remaining credits

---

## ⚙️ Configuration

### Model Selection

You can modify which models are used in the code:

**Text-to-Cypher Model** (in `backend/text_to_cypher_v2.py`):
```python
model: str = "qwen/qwen3-coder:free"  # Change to any OpenRouter model
```

**Response Generator Model** (in `backend/response_generator_v2.py`):
```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Change to any HuggingFace model
```

### Environment Variables

- **`HF_HOME`**: Set the cache location for HuggingFace models
  ```bash
  export HF_HOME=/path/to/cache
  ```

### Using Different Versions

The project includes multiple versions of components:

- **`text_to_cypher.py`**: Basic version
- **`text_to_cypher_v2.py`**: Improved with OpenRouter (currently used in `app.py`)
- **`text_to_cypher_v3.py`**: Alternative implementation

To switch versions, modify the import in `app.py`:
```python
from backend.text_to_cypher_v2 import TextToCypher  # Change v2 to v3
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. "Connection to Neo4j failed"
- ✅ Verify `database_uri`, `username`, and `password` in `config.toml`
- ✅ Ensure your Neo4j instance is running
- ✅ Check network/firewall settings (especially for AuraDB)

#### 2. "OpenAI API Error" or "OpenRouter API Error"
- ✅ Verify your `openai_api_key` in `config.toml`
- ✅ Check if you have API credits remaining
- ✅ Ensure you're using the correct API key format (`sk-or-v1-...`)

#### 3. "No data found from query"
- ✅ Verify the database dump was loaded correctly
- ✅ Check if the graph contains data by running: `MATCH (n) RETURN count(n)` in Neo4j Browser
- ✅ Review the generated Cypher query for correctness

#### 4. "Model download timeout" (for local models)
- ✅ Ensure stable internet connection
- ✅ Set `HF_HOME` to a directory with sufficient space
- ✅ Manually pre-download models using:
  ```bash
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"
  ```

#### 5. Cypher query syntax errors
- ✅ Check that the generated query follows Neo4j Cypher syntax
- ✅ The model may need better prompting or few-shot examples
- ✅ Verify schema alignment - the model should only use labels/relationships from `schema.txt`


## 📝 Additional Notes

### Project Background

This system demonstrates a practical application of knowledge graphs for question answering. By representing the Catan board game as a graph with nodes (Players, Pieces, Resources, etc.) and relationships (OWNS, HAS_RESOURCE, etc.), we can answer complex questions that would be difficult with traditional databases.

### Performance Considerations

- **Local Response Generator**: Runs on CPU by default (Qwen2.5-0.5B-Instruct is lightweight)
- **Text-to-Cypher**: Uses cloud API for better accuracy
- **Trade-off**: Local models are free but less accurate; cloud models cost money but perform better