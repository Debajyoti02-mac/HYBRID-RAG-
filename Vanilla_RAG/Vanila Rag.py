# Load Data 
import warnings 
warnings.filterwarnings('ignore')
from langchain_community.document_loaders import CSVLoader 
loader = CSVLoader(file_path="qoute_dataset.csv")
pages = loader.load()
len(pages)

# Create chunks : 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=120)
text = text_spliter . split_documents(pages)
chunks=[c.page_content for c in text]
metadatas = [c.metadata for c in text]
len(chunks)
# Vector Database 
import chromadb 
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_function= SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./path")
collection = client.get_or_create_collection(name="qoute",embedding_function=embedding_function)

if collection.count() == 0:
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
        metadatas=metadatas
    )
print(collection.count())

from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq 
Groq = ChatGroq(model="llama-3.1-8b-instant",temperature=0,api_key=key)

from langchain_core.tools import tool 
import ast
@tool 
def calculator(expression:str):
    """ Do the calculation based on the impression """
    return str(ast.literal_eval(expression))

    
@tool 
def contextual_Function(query:str)->str:
    """ provide the answers based on the context , dont generalize outside answers without the provided context """
    result = collection.query(query_texts=[query] , n_results=5)
    document = result[ 'documents'][0]
    distance = result["distances"][0]
    
    
    thresold = 1.9
    depend_li=[]
    for doc , i  in zip(document , distance ):
        
        if thresold > i : 
            depend_li.append(doc)
        
    if not depend_li:
        return "NOT RELEVENT CONTENT" 
    return "\n\n".join(depend_li)
    
    
prompt =f""" 
Your are a reliable Assistent ,
Provide answers based in the codex or provided csv , 
stick to the content dont try to provide outside knowledge 

follow the instructions : 

1. provide answers only using provided CSV context 
2. if content dosent content the answers then simply call "i dont know anything"
"""

tool_list=[calculator , contextual_Function]
tool_name={t.name : t for t in tool_list}
Groq_tool = Groq.bind_tools(tools=tool_list)

def tools_function(question : str):
    content = contextual_Function.invoke({"query":question})
    if content == "NOT RELEVENT CONTENT":
        return "NOT RELEVENT CONTENT"
    else :
        massage=[{
            "role":"system",
            "content":prompt
        },
                 {
                     "role":"user",
                     "content":f"""
                     so answer the provided content 
                     qustion : {question} , 
                     content : {content}
                     
                     """
                 }
                   ]
    final=Groq.invoke(massage)
    return final.content





qustion = "'The world as we have created it is a process of our thinking' next line  ?"
answer = tools_function(qustion)
print(answer)

