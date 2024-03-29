# @author : Defne Demirtuerk
# Python Chatbot application with data augmentation.
# The chatbot application is augmented with several recent newspaper articles.

# necessary libraries
import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_data_folder():
    """Helper function to create a new folder.
    """
    if not os.path.exists("news_data"):
        os.makedirs("news_data")

def scrape_url(url):
    """Scrape information from URLs using BeautifulSoup library.
    Args:
        url: URL of a news article.
    Returns: the content of the webpage.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    else:
        print(f"Failed to retrieve content from {url}")
        return None

def save_to_file(data, filename):
    """Save information from URLs to .txt files.
    Args:
        data: Content of a news article.
        filename: File to be saved.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(data)

def scrape_and_save_urls_to_files(input_file):
    """Scrape method to get information from newspaper sites.
    Args:
        input_file: .txt file that stores the URLs of news articles.
    Returns: creates .txt files with the content it reads from the URLs.
    """
    create_data_folder()

    with open(input_file, 'r') as file:
        urls = file.read().splitlines()

    for url in urls:
        content = scrape_url(url)
        if content is not None:
            filename = f"news_data/{url.split('//')[1].replace('/', '_')}.txt"
            save_to_file(content, filename)
            print(f"Scraped and saved content from {url} to {filename}")


def load_docs(directory):
    """Load text data for further processing of LLM.
    Args:
        directory: file directory that stores the .txt files of news articles.
    Returns: 
        documents: combined information of .txt files.
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=500, chunk_overlap=50):
    """Splitter method to divide data into chunks.
    Args:
        documents: combined information of .txt files.
        chunk_size: text chunk size.
        chunk_overlap: chunk overlap size.
    Returns: 
        docs: splitted chunks of text data.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)
    return docs


def get_answer(query, history, db, chain):
    """Method to retrieve the answer of LLM for the provided query.
    Args:
        query: Query from the user.
        history: chat history for the agent.
        db: vectore-store object.
        chain: q&a chain to get response for the query
    Returns: 
        answer: the response of the LLM.
    """
    similar_docs = db.similarity_search(query, k=2) # get two closest chunks
    prompt = ChatPromptTemplate.from_messages(messages=history)
    answer = chain.run(input_documents=similar_docs, question=prompt)
    return answer


def main():

    # set up directories to 
    current_directory = os.path.dirname(os.path.abspath(__file__))
    news_url_directory = os.path.join(current_directory, 'news_urls.txt')
    scrape_and_save_urls_to_files(news_url_directory)

    # start setting up the streamlit UI
    st.title(':red[AI Assistant] ☕')

    # Create an empty placeholder for API key input
    api_key_placeholder = st.empty()
    st.session_state.api_key = api_key_placeholder.text_input('OpenAI API Key', type='password',value=None)

    # ask open ai key from user
    if st.session_state.api_key:
        # Remove the API key input element
        api_key_placeholder.empty()
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        success =st.success("API Key is provided. Loading other functionalities.")
    else:
        st.warning('Please enter your OpenAI API key!', icon='⚠')
        st.stop()

    # use gpt-4 model for ai assistance
    # initializing the embeddings        
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4-1106-preview",temperature=0)
    documents = load_docs("./news_data")
    docs = split_docs(documents)
    #set up vector store
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings
    )
    chain = load_qa_chain(llm)

    # continue set up the streamlit application
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-1106-preview"

    with st.chat_message("ai"):
        st.write("Hello and Welcome to the AI Assistant. How may I help you?")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat messages from history
    for message in st.session_state["messages"]:
        if message[0] != "system":
            with st.chat_message(message[0]):
                st.markdown(message[1])

    prompt = st.chat_input("Ask me anything, please!")
    st.session_state.messages.append(("system","""You are a helpful assistant that answers user questions.
            It is imperative that if you don't know about the answer, you should utilise additional context. 
                                      Do not include mentioning the context in your answer."""))
    # react to user input
    if prompt:
        success.empty() 
        #display user message in chat message container
        with st.chat_message("human"):
            st.markdown(prompt)
        #add user message to chat history
        st.session_state.messages.append(("human", prompt))

        with st.chat_message("ai"):
            message_placeholder = st.empty()
            history = st.session_state.messages
            full_response = get_answer(prompt,history,db,chain) 
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(("ai", full_response))


if __name__ == "__main__":
    main()