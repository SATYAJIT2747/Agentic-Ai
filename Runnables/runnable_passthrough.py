from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough
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
    template="Write a moral lesson about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="explain the following moral lesson in simple words: {text}.",
    input_variables=["text"]
)

parser = StrOutputParser()

lesson_chain = RunnableSequence(prompt1, model1, parser)
parallel_chain = RunnableParallel({"lesson": RunnablePassthrough(), "explanation": RunnableSequence(prompt2, model2, parser)})
final_chain = RunnableSequence(lesson_chain, parallel_chain)
# final_chain.invoke({"topic": "honesty"})
result = final_chain.invoke({"topic": "honesty"})
print(result)
