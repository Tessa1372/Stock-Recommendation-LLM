# Import necessary libraries
import os
import time
from bs4 import BeautifulSoup
import re
import requests
import base64
import json
import yfinance as yf
import langchain
from langchain.agents import Tool, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain_groq import ChatGroq
import warnings
warnings.filterwarnings("ignore")

st.header('Stock Recommendation System with Groq')

# API Key input for Groq
groq_api_key = st.sidebar.text_input('Groq API Key', type='password')

st.sidebar.write('This tool provides recommendation based on the RAG & ReAct Based Schemes:')
lst = ['Get Ticker Value',  'Fetch Historic Data on Stock','Get Financial Statements','Scrape the Web for Stock News','LLM ReAct based Verbal Analysis','Output Recommendation: Buy, Sell, or Hold with Justification']

s = ''
for i in lst:
    s += "- " + i + "\n"
st.sidebar.markdown(s)

# Main execution if Groq API key is provided
if groq_api_key:
    # Initialize Groq-based LLM (replace ChatOpenAI with the equivalent Groq method)
    # You may need to use GroqFlow, Groq API, or relevant package
    # Assuming Groq has similar temperature and model settings
    llm = ChatGroq(temperature=0, model_name='mixtral-8x7b-32768', groq_api_key=groq_api_key)

    # Function to get historical stock prices
    def get_stock_price(ticker):
        if "." in ticker:
            ticker = ticker.split(".")[0]
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        df = df[["Close","Volume"]]
        df.index=[str(x).split()[0] for x in list(df.index)]
        df.index.rename("Date",inplace=True)
        return df.to_string()

    # Function to format Google search query
    def google_query(search_term):
        if "news" not in search_term:
            search_term = search_term+" stock news"
        url = f"https://www.google.com/search?q={search_term}"
        url = re.sub(r"\s","+",url)
        return url

    # Function to scrape recent stock news
    def get_recent_stock_news(company_name):
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
        g_query = google_query(company_name)
        res=requests.get(g_query,headers=headers).text
        soup = BeautifulSoup(res,"html.parser")
        news=[]
        for n in soup.find_all("div","n0jPhd ynAwRc tNxQIb nDgy9d"):
            news.append(n.text)
        for n in soup.find_all("div","IJl0Z"):
            news.append(n.text)

        if len(news) > 6:
            news = news[:4]
        else:
            news = news
        
        news_string = ""
        for i,n in enumerate(news):
            news_string += f"{i}. {n}\n"
        top5_news = "Recent News:\n\n" + news_string
        
        return top5_news

    # Function to get financial statements
    def get_financial_statements(ticker):
        if "." in ticker:
            ticker = ticker.split(".")[0]
        company = yf.Ticker(ticker)
        balance_sheet = company.balance_sheet
        if balance_sheet.shape[1] > 3:
            balance_sheet = balance_sheet.iloc[:,:3]
        balance_sheet = balance_sheet.dropna(how="any")
        balance_sheet = balance_sheet.to_string()
        return balance_sheet

    # Initialize DuckDuckGo search engine
    search = DuckDuckGoSearchRun()
    tools = [
        Tool(
            name="Stock Ticker Search",
            func=search.run,
            description="Use only when you need to get stock ticker from internet, you can also get recent stock related news."
        ),
        Tool(
            name="Get Stock Historical Price",
            func=get_stock_price,
            description="Use when you are asked to evaluate or analyze a stock. This will output historic share price data. You should input the stock ticker to it."
        ),
        Tool(
            name="Get Recent News",
            func=get_recent_stock_news,
            description="Use this to fetch recent news about stocks."
        ),
        Tool(
            name="Get Financial Statements",
            func=get_financial_statements,
            description="Use this to get financial statement of the company. With the help of this data company's historic performance can be evaluated."
        )
    ]

    # Initialize agent with Groq-based LLM and tools
    zero_shot_agent = initialize_agent(
        llm=llm,
        agent="zero-shot-react-description",
        tools=tools,
        verbose=True,
        max_iteration=4,
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )

    # Stock prompt modification
    stock_prompt = """You are a financial advisor. Provide stock recommendations for the given query.
Answer the questions as best as you can. You have access to the following tools:

- Get Stock Historical Price: Use when analyzing a stock.
- Stock Ticker Search: Use only when you need the stock ticker.
- Get Recent News: Fetch the latest news about stocks.
- Get Financial Statements: Fetch the financial statement of the company.

Steps to follow:
1. Identify the company name and search for the "company name + stock ticker."
2. Use "Get Stock Historical Price" to get historical prices.
3. Use "Get Financial Statements" to get financial data.
4. Use "Get Recent News" to search for stock-related news.
5. Analyze the stock and give a detailed recommendation: Buy, Hold, or Sell.

Use the following format:

Question: {input}
Thought: {agent_scratchpad}
Action: [Mention tool to use]
Action Input: [Input for the tool, e.g., company name or stock ticker]
Observation: [Result from the tool]
Final Answer: [Provide stock recommendation]
Begin!"""

    zero_shot_agent.agent.llm_chain.prompt.template = stock_prompt

    # User input in chat
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = zero_shot_agent(f'Is {prompt} a good investment choice right now?', callbacks=[st_callback])
            st.write(response["output"])
