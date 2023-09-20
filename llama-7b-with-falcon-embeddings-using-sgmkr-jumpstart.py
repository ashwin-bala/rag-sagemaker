from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
import langchain

# langchain.verbose = True
# langchain.debug = True

import sys
import boto3
import json
import redis
from langchain.vectorstores.redis import Redis
import os

import sagemakerembeddingsclient
sageMaker_embeddings = sagemakerembeddingsclient.getSageMakerEmbeddings()


region = "us-east-1"
endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5

redis_host = os.environ.get("REDIS_HOST")
redis_pwd = os.environ.get("REDIS_PWD")
redisConn = f"redis://:{redis_pwd}@{redis_host}:6379"

vector_schema = {
    "algorithm": "HNSW"
}

def getRedisVectorStore():
    rds = Redis.from_existing_index(
        sageMaker_embeddings,
        index_name="idx",
        redis_url=redisConn,
        schema=vector_schema
    )
    return rds

prompt_template = """
  The following is a friendly conversation between a human and an AI. 
  The AI is talkative and provides lots of specific details from its context.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  {context}
  Instruction: Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
  if not present in the document. 
  Solution:"""
  
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

condense_qa_template = """
Given the following conversation and a follow up question, rephrase the follow up question 
to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode('utf-8')
        
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
      #  print("---------------------")
        print(response_json)
        return response_json[0]["generation"]

content_handler = ContentHandler()

llm=SagemakerEndpoint(
        endpoint_name=endpoint_name, 
        region_name=region, 
        content_handler=content_handler,        
        model_kwargs={"max_new_tokens": 64, "top_p": 0.9, "temperature": 0.6, "return_full_text": False},
        endpoint_kwargs = {"CustomAttributes":"accept_eula=true"}
    )
  

def chainWithLLM(llm,query,cust_temp):

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    llm.model_kwargs = {'temperature': cust_temp}
    print(llm)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=getRedisVectorStore().as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa({"query": query})
    return result
    # print(result['result'])
    # print(result['source_documents'])

result = chainWithLLM(llm, "what is continuos integration?", 0.8)
print(result['result'])