import re

from langchain_core.documents import Document 

def clean_text(text: str) -> str:
    """
    Clean the text by removing unwanted characters and formatting.
     """


    text = text.replace("\x7f", "\n- ")
    text = text.replace("\ufffd", "-")
    text = text.replace("�", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")

    #Remove extra spaces and tabs within each line 
    text = re.sub(r"[\t]+", " ",text)

    #Nomrmalize excessive blank lines to a maaximum of two lines
    text = re.sub(r"\n{3,}", "\n\n", text)


    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Clean a list of documents by applying text cleaning to each document's page_content.
    """

    cleaned_documents = []

    for document in documents:
        cleaned_content = clean_text(document.page_content)
        cleaned_document = Document(
            page_content=cleaned_content,
            metadata=document.metadata,
        )
        cleaned_documents.append(cleaned_document)

    return cleaned_documents
