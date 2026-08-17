from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

model1 = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.7
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.7
)

prompt1 = PromptTemplate(
    template="Write a short note on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Write a short story about a {animal}.",
    input_variables=["animal"]
)

parser = StrOutputParser()

# chain = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel({"note": RunnableSequence(prompt1, model1, parser), "story": RunnableSequence(prompt2, model2, parser)})

result = parallel_chain.invoke({"topic": "AI", "animal": "cat"})
print(result)