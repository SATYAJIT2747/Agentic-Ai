from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0
)
text = "This is a sample text that needs to be split into smaller chunks for processing."
chunks = splitter.split_text(text)
