# document-based-qna

This project implements a document-based question answering system inspired by recent research advances. It combines strong retrieval methods with large language models (LLMs) to provide accurate answers from complex documents. The system uses document structure—such as sections, tables, and figures—alongside semantic chunking and retrieval-augmented generation (RAG). Documents are divided into meaningful chunks, which, along with user queries, are embedded using transformer models. Metadata like page numbers and section titles help guide retrieval. Relevant chunks are selected based on embedding similarity, and an LLM generates answers grounded in the retrieved context. 

Methodology:

To implement the system, I have combined the methodologies from Saad-Falcon et al. [2024] and
Muludi et al. [2024]. The methodology is demonstrated below -

![image](https://github.com/user-attachments/assets/305b7d20-deba-4b83-8c2b-b05a8da62461)


• **Text Extraction:** The system uses the PyMuPDF (fitz) library to read the PDF document. For
each page, it performs chunking—where each chunk typically corresponds to a paragraph—to
maintain the semantic meaning of the content. It also identifies section titles using regular
expressions. Each chunk is then associated with its respective page number and section title.

• **Metadata Construction:** The metadata is created by collecting all unique page numbers and
section titles from the extracted chunks. This metadata helps in categorizing and retrieving
relevant information based on user queries.

• **Semantic Embedding Generation:** The sentence-transformers library (specifically, the ’all-
MiniLM-L6-v2’ model) is used to convert each text chunk into a vector embedding.
• User Query Processing: When a user submits a query, it is also converted into an embedding
using the same model.

• **Query Classification:** A large language model (LLM) classifies the user’s query as referring
to a page number, section, or a general concept, using document metadata and a prompt
template. The LLM analyzes both the content and structure of the query in the context of the
document’s metadata to determine the most appropriate retrieval strategy.

• **Context Retrieval via ’get’ Functions:** Based on the function call initiated by the LLM, the
following functions will be called:

– get top chunk: The cosine similarity between the query embedding and each chunk
embedding is calculated to find the most relevant content.The top-N most chunks are
returned.

– get page chunks: Retrieves and concatenates all chunks from the specified page number.

– get section chunks: Retrieves and concatenates all chunks from the specified section
title.

• **Answer Generation:** The large language model (LLM) receives the retrieved context and the
user’s question, and generates an answer strictly based on the provided context.


**Technology Stack** The code uses the all-MiniLM-L6-v2 transformer model from Sentence Trans-
formers to generate semantic embeddings for text chunks and user queries. For large language
model (LLM) tasks, the Google Generative AI (Gemini) API via LangChain is utilized. This LLM
API was chosen because it was free, however it does not support direct function calling or tool
use from within the model and hence the complete methdology of PDFTriage was not used. The
frontend is developed using Flask.
