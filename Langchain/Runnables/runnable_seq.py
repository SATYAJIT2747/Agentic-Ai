from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.7
)

prompt = PromptTemplate(
    template="Write a short story about a {animal}.",
    input_variables=["animal"]
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)

result = chain.invoke({"animal": "cat"})
print(result)