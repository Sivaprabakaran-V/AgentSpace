from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import (create_react_agent, AgentExecutor)
from langchain import hub
from tools.tools import get_profile_url_tavily

import os


def lookup(name:str) -> str:
    llm = ChatGoogleGenerativeAI( model="gemini-1.5-flash",  
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0)
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""

    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"]  )

    tools_for_agent = [Tool( name = "Crawl Google 4 linkedin profile page",
                          func= get_profile_url_tavily,
                          description="useful for when you need get the Linkedin Page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke({"input": prompt_template.format_prompt(name_of_person=name) }
    )

    linkedin_url = result['output']
    return linkedin_url

if __name__ == "__main__":
    print(lookup(name="Allie Miller"))