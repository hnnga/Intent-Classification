from pymilvus import Collection
from flow.nodes.node1_input import get_embedding
from milvus_utils.milvus import COLLECTION_NAME

SIM_THRESHOLD = 0.9

def check_vector_db(user_question):
    embed = get_embedding(user_question)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    results = collection.search(
        data=[embed],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=1,
        output_fields=["question", "intent"])

    matches = results[0]
    if matches and matches[0].score > SIM_THRESHOLD:
        matched = matches[0]
        return {
            "found": True,
            "intent": matched.entity.get("intent"),
            "matched_question": matched.entity.get("question")}

    return {"found": False}
