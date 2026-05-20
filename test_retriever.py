from retriever import SimpleRetriever

retriever = SimpleRetriever()

query = "What does an abnormal chest X-ray indicate?"
context = retriever.retrieve(query)

print("\nRetrieved context:\n")
print(context)