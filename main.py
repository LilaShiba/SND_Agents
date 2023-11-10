from utils.agent import Agent

if __name__ == "__main__":

    embedding_params = ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.7]
    path = 'documents/meowsmeowing.pdf'
    db_path = 'chroma_db/agent_snd'

    testAgent = Agent('agent_snd', db_path, 0, embedding_params, True)
    # testAgent.add_memory("documents/LtoA.pdf")
    testAgent.start_chat()
