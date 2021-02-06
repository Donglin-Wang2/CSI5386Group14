import logging
import pandas as pd
from six import iteritems
from web.datasets.analogy import fetch_wordrep, fetch_google_analogy, fetch_msr_analogy, fetch_semeval_2012_2
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW, fetch_TR9856
from web.embeddings import fetch_GloVe, fetch_LexVec, fetch_HDC, fetch_conceptnet_numberbatch, fetch_SG_GoogleNews
from web.evaluate import evaluate_similarity, evaluate_analogy, evaluate_on_semeval_2012_2, evaluate_on_WordRep

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# Fetch GloVe embedding (warning: it might take few minutes)
embeddings = {
    "GloVe": fetch_GloVe(corpus="wiki-6B", dim=300),
    "SG_GoogleNews": fetch_SG_GoogleNews(),
    "LexVec": fetch_LexVec(),
    "HDC": fetch_HDC(),
    "Comceptnet": fetch_conceptnet_numberbatch()
}

# Define tasks
# datasets = {
#     "MTurk": fetch_MTurk(),
#     "MEN": fetch_MEN(),
#     "WS353": fetch_WS353(),
#     "RG65": fetch_RG65(),
#     "Rare Words": fetch_RW(), 
#     "SIMLEX999": fetch_SimLex999(),
#     "TR9856": fetch_TR9856()
# }

analogy_datasets = {
    "MSR_WordRep": None,
    "Google_Analogy": fetch_google_analogy(),
    "MSR": fetch_msr_analogy(),
    "SEMEVAL 2012 Task 2": None
}

result_analogy = pd.DataFrame(columns=embeddings.keys(), index=analogy_datasets.keys())

for embedding_name, embedding in embeddings.items():
    for dataset_name, data in iteritems(analogy_datasets):
        if dataset_name == "SEMEVAL 2012 Task 2":
            result_analogy[embedding_name][dataset_name] = evaluate_on_semeval_2012_2(embedding)
        elif dataset_name == "MSR_WordRep":
            result_analogy[embedding_name][dataset_name] = evaluate_on_WordRep(embedding)
        else:
            result_analogy[embedding_name][dataset_name] = evaluate_analogy(embedding, data.X, data.y)

result_analogy.to_csv('./embedding_results_analogy.csv')

# result = pd.DataFrame(columns=embeddings.keys(), index=datasets.keys())

# for embedding_name, embedding in embeddings.items():
#     for dataset_name, data in iteritems(datasets):
#        result[embedding_name][dataset_name] = evaluate_similarity(embedding, data.X, data.y)

# result.to_csv('./embedding_results.csv')
