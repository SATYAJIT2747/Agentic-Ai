from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.7
)
loader = TextLoader("C:\\Users\\satya\\Desktop\\Agentic-Ai\\doc.txt")
text = loader.load()

prompt = PromptTemplate(
    template="Write a summary of the following text: {text}",
    input_variables=["text"]
)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text': text[0].page_content})
print(result)