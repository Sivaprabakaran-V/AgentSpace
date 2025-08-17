from re import search
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
from third_parties.linkedin_search import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    summary_template = """Given the Linkedin information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template
    )

    # Use Google Gemini (Free via Google AI Studio)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # or "gemini-1.5-pro" for higher quality
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    chain = summary_prompt_template | llm | StrOutputParser()
    linkedin_data = scrape_linkedin_profile(linkedin_url="https://www.linkedin.com/in/alliekmiller/")

    res = chain.invoke(input={"information": linkedin_data})
    print(res)

    print("This is the end of the chaining single prompt example.")
