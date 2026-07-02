from app.retriever import retriever

results = retriever.search(
    "Java developer with stakeholder communication",
    top_k=5
)

for r in results:
    print("=" * 60)
    print(r["name"])
    print(r["url"])
    
    
    