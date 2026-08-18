from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)
text = "This is a sample text that needs to be split into smaller chunks for processing."
chunks = text_splitter.split_text(text)