from utils.pack import Pack
from utils.agent import Agent
from utils.metrics import ThoughtDiversity
from collections import defaultdict
import networkx as nx


if __name__ == "__main__":

    learning_to_act = "chroma_db/agent_ltoa"
    system_neural_diversity = "chroma_db/agent_snd"
    foundational_models = "chroma_db/agent_foundation"
    norbet_cog = "chroma_db/agent_norbert"
    viz_quant = "chroma_db/agent_quant"
    cot_path = "chroma_db/agent_cot"

    embedding_params = [
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9],
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.1],
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.5],
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9],
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.1],
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.5],
        ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.8]
    ]
    # name, path, cot_type, new_bool
    agent_specs = [
        ['agent_ltoa', learning_to_act, 0, True],
        ['agent_snd', system_neural_diversity, 0, True],
        ['agent_foundation', foundational_models, 0, True],
        ['agent_quant', viz_quant, 0, True],
        ['agent_norbert', norbet_cog, 0, True],
        ['agent_cot', cot_path, 0, True]
    ]

    test_pack = Pack(agent_specs, embedding_params)
    edges = test_pack.update_weighted_edges(
        question="Imagine how a neuron for a neural network may be reimagined based on the text.",  k=3)
    print(edges)
