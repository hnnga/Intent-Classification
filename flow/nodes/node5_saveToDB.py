from flow.nodes.node1_input import get_embedding
from milvus_utils.milvus import insert_data

def save_question_and_paraphrases(original, paraphrases, intent):

    # store all original question + paraphrases
    all_questions = [original] + paraphrases
    embeddings = [get_embedding(q) for q in all_questions]
    insert_data(all_questions, embeddings, intent)
