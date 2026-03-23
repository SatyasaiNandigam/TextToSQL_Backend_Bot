from schema_extractor_advanced import smart_retriever


query = input("Enter your query: ")

docs = smart_retriever(query, k_fetch=12, k_return=4)
tables = [doc.metadata["table_name"] for doc in docs]

print("Relevant tables:", tables)

print([doc.page_content for doc in docs])