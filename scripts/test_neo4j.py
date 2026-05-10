from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "10451045"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

try:
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j connected' AS msg")
        print(result.single()["msg"])
finally:
    driver.close()