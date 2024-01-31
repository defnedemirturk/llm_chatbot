# AI assistant

## Introduction
------------
The AI Assistant is a Python chatbot application to which you can ask questions, and the application will provide relevant responses based on the content of your questions. With its web-scraping functions, the chatbot is augmented with news information from several news webpages. This app utilizes a language model to generate accurate answers to your queries. (Please note that the current version of the app will only respond to questions related to the loaded news articles aside from the information stored on OpenAI servers.)

## How It Works
------------

The application follows these steps to provide responses to your questions:

1. The app scrapes recent news from news webpages using BeautifulSoup framework and saves their text content as txt files.

2. The information is loaded from .txt files and is divided into smaller text chunks to be processed.

3. The application uses a language model to generate embeddings of the text chunks.

4. The text chunks are passed to the language model to generate a response based on the relevant content of the txt files.

## Dependencies and Installation
----------------------------
To install the AI assistant, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI. You will be asked to provide it to the AI assistant UI while entering the application.

## Usage
-----
To use the AI assistant, follow these steps:

1. Ensure that you have installed the required dependencies.

2. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. You will be asked to provide your OpenAI API key at the beginning. 

5. Once the key is validated, you can start asking questions using the chat interface.# llm_chatbot
