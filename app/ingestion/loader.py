from pypdf import PdfReader
from llama_index.core.node_parser import SentenceSplitter
import os
BASE_DIR = os.path.abspath(os.path.curdir)
DEFAULT_PATH = os.path.join(os.path.abspath(os.path.join(BASE_DIR,"..")),"data","raw")

text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

def load_pdfs(path=DEFAULT_PATH, max_pages=30):
    documents = []
    try:
        for file in os.listdir(path):
            if file.endswith(".pdf"):
                file_path = os.path.join(path, file)
                reader = PdfReader(file_path)

                total_pages = reader.pages
                if len(total_pages) > max_pages:
                    continue

                title = (
                    reader.metadata.title
                    if reader.metadata and reader.metadata.title
                    else file
                )

                for page_number, page in enumerate(total_pages):
                    text = page.extract_text()

                    if not text:
                        continue

                    chunks = text_parser.split_text(text)

                    for chunk in chunks:
                        documents.append(
                            {
                                "text": chunk,
                                "metadata": {
                                    "source": file,
                                    "title": title,
                                    "page": page_number,
                                },
                            }
                        )

        return documents

    except Exception as e:
        print(e)
        