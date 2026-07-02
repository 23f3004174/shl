import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading catalog...")

with open("data/shl_product_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

documents = []
metadata = []

for item in catalog:

    text = f"""
Name: {item.get("name","")}

Description:
{item.get("description","")}

Job Levels:
{', '.join(item.get("job_levels",[]))}

Languages:
{', '.join(item.get("languages",[]))}

Categories:
{', '.join(item.get("categories",[]))}

Remote:
{item.get("remote_testing","")}

Adaptive:
{item.get("adaptive_irt","")}

Duration:
{item.get("assessment_length","")}
"""

    documents.append(text)

    metadata.append({
        "name": item.get("name"),
        "url": item.get("link"),
        "description": item.get("description"),
        "job_levels": item.get("job_levels"),
        "languages": item.get("languages"),
        "categories": item.get("categories")
    })

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

faiss.write_index(index, "data/faiss.index")

with open("data/catalog_processed.json","w",encoding="utf-8") as f:
    json.dump(metadata,f,indent=2,ensure_ascii=False)

print("="*60)
print("INDEX CREATED")
print("Assessments:",len(metadata))
print("="*60)