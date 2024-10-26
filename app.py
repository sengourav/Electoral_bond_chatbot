from langchain.chains import LLMChain
import chainlit as cl
import os
import openai
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory

# from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager

from langchain.agents import initialize_agent, Tool

from langchain.chains import LLMMathChain
from langchain.utilities import GoogleSerperAPIWrapper

# from langchain.tools.retriever import create_retriever_tool




# Define constants for model and vector store paths
# MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
openai_api_key = os.environ.get("OPENAI_API_KEY")
serper_api_key=os.environ.get("SERP_API_KEY")

VECTOR_STORE_PATH = "Downloads/dsp/NLP/vector_faiss/vector_faiss"
# EMBEDDING_MODEL_NAME = "avsolatorio/GIST-large-Embedding-v0"
# os.environ['OPENAI_API_KEY'] =os.environ.get("OPEN_API_KEY")
openai.api_key = openai_api_key
os.environ["SERPER_API_KEY"]=serper_api_key



@cl.cache
def instantiate():
# Instantiate LlamaCpp model
    # llm = LlamaCpp(
    #     streaming=True,
    #     model_path=MODEL_PATH,
    #     max_tokens=700,
    #     temperature=0.7,
    #     top_p=1,
    #     n_gpu_layers=-1,
    #     n_batch=512,
    #     n_ctx=16000,
    #     f16_kv=True,
    #     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    #     verbose=True
    # )

    # # Create embeddings
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME,
    #     model_kwargs={'device': 'cpu'},
    #     encode_kwargs={'normalize_embeddings': True}
    # )
    llm = ChatOpenAI()
    

    embeddings = OpenAIEmbeddings(show_progress_bar=True)

    # Load vector store
    # index=FAISS.load_local('https://drive.google.com/file/d/1-DgmC8rYns-IRmIGZZF1TcaHztX2i7vq/view?usp=sharing', embeddings, allow_dangerous_deserialization=True)
    index=FAISS.load_local('Downloads/dsp/NLP/vector_faiss/vector_faiss', embeddings,allow_dangerous_deserialization=True)
    return llm , index

llm , vector_store= instantiate()

llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
# Set up tools

search = GoogleSerperAPIWrapper()



tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
]




# Set up agent
agent = initialize_agent(
    tools,
    llm,
    agent="conversational-react-description",
    verbose=True,
    memory=ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", k=5),
    return_intermediate_steps=True,
)


# Set up conversation chain
@cl.on_chat_start
def main():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a News critic."}],
    )


    prompt_template = """Answer the question as News Critic
    Context: {context}


    {chat_history}
    Human: {question}
    Assistant:"""

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template,
        # partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="question", k=5)
    handler = StdOutCallbackHandler()
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    cl.user_session.set("llm_chain", llm_chain)
    
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 10}
        )
    
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    retriever= cl.user_session.get("retriever")
    agent = cl.user_session.get("agent")
    
    # Get the user's message
    user_input = message.content
    docs = retriever.get_relevant_documents(user_input)
    msg = cl.Message(content="")
    await msg.send()
    # Generate response using LLMChain
    response = await llm_chain.arun(question=user_input, context=docs, callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)],agent=agent)
    # response=await cl.make_async(agent.run)(message.content, callbacks=[cl.LangchainCallbackHandler(stream_final_answer=True)])

    
    for part in (response):
        if token := part or "":
            await msg.stream_token(token)
    # # Send the response back to the user
    # await message.respond(response)
    await msg.update()

# @cl.on_stop
# def on_stop():
#     print("The user wants to stop the task!")