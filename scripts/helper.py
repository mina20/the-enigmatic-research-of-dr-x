from llama_index.core import Document

def get_llama_documents(pages: list) -> list[Document]:
    """
    Convert extracted text pages into LlamaIndex Document format.
    
    Args:
        pages (list): A list of dicts with keys:
            - 'text': str
            - 'file_name': str
            - 'file_type': str
            - 'page_or_sheet': str

    Returns:
        List[Document]: LlamaIndex-compatible Document objects
    """
    llama_docs = []
    for page in pages:
        llama_doc = Document(
            text=page['content'],
            metadata={
                "file_name": page["file_name"],
                "file_type": page["file_type"],
                "page_or_sheet": page["page_or_sheet"]
            }
        )
        llama_docs.append(llama_doc)
    
    return llama_docs
