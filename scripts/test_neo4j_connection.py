from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("RETURN 'Kết nối Neo4j thành công!' AS message")
    print(result.single()["message"])
