from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

if __name__ == "__main__":
    load_dotenv()

    print("Chaining single prompt example:"
          " This example demonstrates how to use a single prompt with LangChain.")
    information ="""Luka Modrić, born on 9 September 1985 in Zadar, Croatia, is a Croatian professional footballer and captain of the national team, widely regarded as one of the greatest midfielders of his generation. Renowned for his vision, passing, and composure, he began his career at Dinamo Zagreb, moved to Tottenham Hotspur in 2008, and joined Real Madrid in 2012, where he became a pivotal figure in winning multiple UEFA Champions League and La Liga titles. He led Croatia to the 2018 FIFA World Cup final, winning the Golden Ball, and that same year claimed the Ballon d’Or, breaking the Messi–Ronaldo dominance. His leadership, work rate, and technical mastery have kept him performing at the highest level well into his late 30s."""
    summary_template = """given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them"""

    summary_prompt_template= PromptTemplate(input_variables=["information"], template=summary_template)

    llm= ChatOllama(model="llama3")

    chain = summary_prompt_template | llm | StrOutputParser()


    res = chain.invoke(input = {"information": information})
    print(res)
    print("This is the end of the chaining single prompt example.")