# This code is divided into the following parts:
# 1. Reading the PDF and dividing it into chunks + storing the chunks with metadata
# 2. Converting the chunks using vector embeddings
# 3. Converting the query into vector embeddings 
# 4. Query Classification using LLM
# 4. Calculating the cosine similarity of the chunks and query
# 5. Feeding the query and retrived document to LLM
# 6. Displaying the results


#Imports
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os


#Setting up global variables
database=[]
metadata=[]
model = SentenceTransformer('all-MiniLM-L6-v2')
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    google_api_key=os.getenv('google_api_key') 
)

# Code Part 1: Reading and Storing the PDF file
def ingest_pdf(doc):
    doc = fitz.open(doc)
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks") 
        current_title = None
        for block in blocks:
            text = block[4].strip()
            if bool(re.match(r'^\d+\s*\n', text.strip())): # This snippet checks if the text is a section title
                current_title = text.replace('\n', ' ').strip()
                continue
            database.append({
                "Page Number": page_num,
                "Section Title": current_title if current_title else "", # Assigns the section title to the chunks till a new title is found
                "Chunk": text
            })

    title=database[0]['Chunk'] # Storing the title of the document

    # Creating metadata
    metadata={}
    metadata['Pages']=[]
    metadata['Sections']=[]
    for i, entry in enumerate(database):
        if entry['Page Number'] not in metadata['Pages']:
            metadata['Pages'].append(entry['Page Number'])
        if entry['Section Title'] not in metadata['Sections']:
            metadata['Sections'].append(entry['Section Title']) 

    # Code Part 2: Converting chunks to vector embeddings
    chunks = [entry["Chunk"] for entry in database]

    # Generate vector embeddings
    embeddings = model.encode(chunks, show_progress_bar=False)

    # Add embeddings back to your database entries
    for entry, emb in zip(database, embeddings):
        entry["Embedding"] = emb.tolist()  # Convert numpy array to list for easy storage

def answer_query(user_query):
    
    # Part 3: Converting the user query to vector embeddings
    query_embedding = model.encode([user_query])[0] 

    # Part 4: Calculating the cosine similarity of the chunks and query
    def get_top_chunk(query_embedding):
        for entry in database:
            similarity = np.dot(
                np.array(query_embedding), 
                np.array(entry['Embedding'])
            ) / (
                np.linalg.norm(np.array(query_embedding)) * np.linalg.norm(entry['Embedding'])
            )
            entry["Similarity"] = similarity

        # Sort entries by similarity (descending)
        sorted_entries = sorted(database, key=lambda x: x["Similarity"], reverse=True)

        top_chunks = [entry['Chunk'] for entry in sorted_entries[:5]]
        
        return top_chunks
        
    # Part 5: Retrieving the content of a given page
    def get_page_chunks(pageNumber):
        page_text=""
        for i, entry in enumerate(database):
            if entry['Page Number']==pageNumber:
                page_text+=entry['Chunk']
        if page_text=="":
            page_text="This page number does not exist."
        return page_text

    # Part 6: Retrieving the content of a given section
    def get_section_chunks(sectionTitle):
        section_text=""
        for i, entry in enumerate(database):
            if sectionTitle.lower() in entry['Section Title'].lower():
                section_text+=entry['Chunk']
        if section_text=="":
            section_text="This section does not exist, please try some different section."
        return section_text

    # Part 7: Using LLM for Query Classification
    template="""

    You are an agent that classifies user questions about a document into one of the following types:

    Rules:
    - If the query refers to any page number, respond with: page number,<page number as a digit>.
    - If the query mentions a section, heading, or title, a part of which is mentioned in the meta data respond with: section,<section name or 'false'>.
    - If the query mentions both a page number and a section, apply the rule for section.
    - For all other queries, respond with: concept.

    Use the provided metadata of the document and user query to understand what rules can be applied.
    Respond with only one line in the following format:
    <function_type_1>,<argument_1>; <function_type_2>,<argument_2>; ...

    User Query: {user_query}
    Meta Data: {metadata}
    Your Response:

    """
    prompt = PromptTemplate(
        input_variables=["user_query","metadata"],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    question_type = chain.run({"user_query": user_query,"metadata":metadata})

    # Part 8: Calling appropriate function based on the query classification
    function_calls = [fc.strip() for fc in question_type.split(";") if fc.strip()]
    context={}
    for func_call in function_calls:
        question_format=func_call.split(",")[0]
        print(question_format)

        if question_format=='page number':
            context[func_call]=get_page_chunks(int(func_call.split(",")[1]))
        elif question_format=='section':
            context[func_call]=get_section_chunks(func_call.split(",")[1])
        else:
            context[func_call]=get_top_chunk(query_embedding)

    # Part 9: Using LLM to answer the question
    template = """You are a question answering agent. Use only the information available in the context to answer the question. 
    If the answer is not available, clearly mention so, do not create your own answers.
    Context:
    {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run({"context": context, "question": user_query})
    return answer



