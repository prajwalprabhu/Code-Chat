import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.utils.math import cosine_similarity
import streamlit as st

from configs import DOCUMENTS_FOLDER, EMBEDDING_MODEL


class RAGRetriever:
    def __init__(
        self,
        vector_store_path="./faiss_index",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        groq_api_key=None,
        groq_model="mixtral-8x7b-32768",
    ):
        """
        Initialize RAG Retriever with FAISS vector store and Groq LLM

        :param vector_store_path: Path to the saved FAISS vector store
        :param embedding_model: Hugging Face embedding model
        :param groq_api_key: Groq API key
        :param groq_model: Groq LLM model to use
        """

        # Set up API key (use environment variable or passed parameter)
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment variable or pass it directly."
            )

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Load vector store

        self.vector_store_path = vector_store_path
        vectorstore = FAISS.load_local(
            self.vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.files: List[str] = list(
            set([x.metadata["source"] for x in vectorstore.docstore._dict.values()])
        )
        del vectorstore
        self.llm = ChatGroq(
            temperature=0.2, model_name=groq_model, groq_api_key=self.groq_api_key
        )

        # Define RAG prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
        You are a helpful AI assistant. Answer the question based only on the following context:

        <context>
        {context}
        </context>

        Question: {question}

        If the context does not contain enough information to answer the question then you can answer it without context
        
        """
        )

    def _create_rag_chain(self, query):
        """
        Create the RAG retrieval and generation chain
        """

        def filter_docs(query, docs):
            files = self.find_relevant_files(query)
            return [doc for doc in docs if doc.metadata["source"] in files]

        try:
            self.vectorstore = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            filtered_docs = filter_docs(query, self.vectorstore.docstore._dict.values())
            if len(filtered_docs) > 5 and len(filtered_docs) < len(self.files):
                del self.vectorstore
                self.vectorstore = FAISS.from_documents(filtered_docs, self.embeddings)
        except Exception as e:
            raise ValueError(f"Error loading vector store: {e}")

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 4,
                "fetch_k": 30,
                "score_threshold": 0.75,
            },  # Return top 4 most similar documents
        )
        # Define the RAG chain
        self.rag_chain = (
            RunnableParallel(
                {"context": self.retriever, "question": RunnablePassthrough()}
            )
            | RunnablePassthrough.assign(
                context=lambda x: "\n\n".join(
                    [doc.page_content for doc in x["context"]]
                )
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve_and_generate(self, query):
        """
        Retrieve relevant documents and generate an answer

        :param query: User's question
        :return: Generated answer
        """
        try:
            self._create_rag_chain(query)
            # Invoke the RAG chain
            response = self.rag_chain.invoke(query)

            # Optional: Print retrieved documents for transparency
            retrieved_docs = self.retriever.invoke(query)
            print("\n--- Retrieved Documents ---")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"Document {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
            del self.vectorstore
            return response

        except Exception as e:
            print(f"Error in retrieval and generation: {e}")
            return "An error occurred while processing your query."

    def find_relevant_files(self, query, threshold=0.7) -> List[str]:
        """Find files relevant to the query."""
        file_names = self.files
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        file_embeddings = [
            embeddings.embed_query(file_name) for file_name in file_names
        ]

        # Embed query
        query_embedding = embeddings.embed_query(query)

        # Compute cosine similarities
        similarities = cosine_similarity([query_embedding], file_embeddings)[0]

        # Filter file names based on similarity threshold
        relevant_files = [
            file_names[i]
            for i in range(len(similarities))
            if similarities[i] >= threshold
        ]
        return relevant_files


def main():
    # Example usage
    try:
        # Initialize RAG Retriever (make sure to set GROQ_API_KEY)
        rag_retriever = RAGRetriever(
            vector_store_path="./faiss_index",
            embedding_model=EMBEDDING_MODEL,
            groq_api_key="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",
            groq_model="llama3-8b-8192",
        )
        # st.title("Code Chat")
        # with st.form("my_form"):
        #     c = st.container()
        #     c1, c2 = c.columns(2)
        #     query = c1.text_area(
        #         "Enter text:", "What are 3 key advice for learning how to code?"
        #     )
        #     submitted = st.form_submit_button("Submit")

        #     if submitted:
        #         response = rag_retriever.retrieve_and_generate(query)
        #         c2.info(response)
        #         print(response)

        while True:
            query = input("Enter text: ")
            print(rag_retriever.find_relevant_files(query, 0.8))
            response = rag_retriever.retrieve_and_generate(query)
            print(response)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
