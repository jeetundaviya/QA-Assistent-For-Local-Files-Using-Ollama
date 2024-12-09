from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_ollama import OllamaLLM,ChatOllama
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

llm = ChatOllama(model="llama3.2",temperature=0)  # Initialize the Ollama model

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str

workflow = StateGraph(state_schema=State)

# Defining the function that calls the model
def model_call(state: State):
    chat_prompt = prompt.invoke(state)
    res = llm.invoke(chat_prompt)
    return {"messages":[res]}

# Defining the single node in the graph
workflow.add_edge(START,"MODEL_CALL")
workflow.add_node("MODEL_CALL",model_call)

# Adding Memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

def setup_langchain_qa(vector_store, model="llama2", retriever_k=3):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retriever_k})
    
    qa_chain = app

    return qa_chain

def query_qs(question:str,vector_store, model="llama2", retriever_k=3):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retriever_k})
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)

    # Combine the documents into a single string
    docs_text = "".join(d.page_content for d in docs)

    input_messages = [HumanMessage(content=question)]
    output = app.invoke({"messages":input_messages,"context":docs_text},config)
    output["messages"][-2].pretty_print()
    output["messages"][-1].pretty_print()

    return output["messages"][-1]