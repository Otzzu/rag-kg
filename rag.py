from database import GraphDatabaseDriver
from response_generator import ResponseGenerator
from text_to_cypher import TextToCypher

with GraphDatabaseDriver() as driver:
    schema = """
Node properties:
- **Player**
- accountId: INTEGER
- username: STRING Example: "Nicki1202"
- **Level**
- id: INTEGER
- name: STRING Example: "OuterSpace"
- **Comment**
- id: INTEGER
- content: STRING Example: "GG! Nice level:)"
Relationship properties:

The relationships:
(:Player)-[:SHARES]->(:Level)
(:Player)-[:SUBMITS]->(:Comment)
(:Level)-[:HAS]->(:Comment)
    """.strip()

    print("Preparing text-to-Cypher pipeline ....")
    ttc = TextToCypher(schema)

    print("Preparing response generator pipeline ....")
    generator = ResponseGenerator(schema)

    interrupt = False
    print("(Interrupt to stop.)")
    while not interrupt:
        try:
            question = input("Question: ")
        except KeyboardInterrupt:
            interrupt = True

        if not interrupt:
            print("Generating Cypher query ....")
            query = ttc(question)
            print(query)

            print("Executing Cypher query ....")
            results = driver.execute_query(query)
            query_result_str = "\n".join([
                str(x) for x in results
            ])
            print(query_result_str)

            print("Generating response ....")
            response = generator(question, query, query_result_str)
            print(response)
    
    print("(Stopped.)")
