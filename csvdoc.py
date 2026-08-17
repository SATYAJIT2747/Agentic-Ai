from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("students.csv")
docs = loader.load()