from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import langchain_core.output_parsers
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    print("Brief Agent starting...")
    information =  """Sir Christopher Edward Nolan (born 30 July 1970) is a British and American filmmaker. Known for his Hollywood blockbusters with structurally complex storytelling, he is considered a leading filmmaker of the 21st century. Nolan's films have earned over $6.6 billion worldwide, making him the seventh-highest-grossing film director. His accolades include two Academy Awards, a Golden Globe Award and two British Academy Film Awards. Nolan was appointed a Commander of the Order of the British Empire in 2019, and received a knighthood in 2024 for his contributions to film. Nolan developed an interest in filmmaking from a young age. After studying English literature at University College London, he made several short films before his feature film debut with Following (1998). Nolan gained international recognition with his second film, Memento (2000), and transitioned into studio filmmaking with Insomnia (2002). He became a high-profile director with The Dark Knight trilogy (2005–2012), and found further success with The Prestige (2006), Inception (2010), Interstellar (2014), and Dunkirk (2017). After the release of Tenet (2020), Nolan parted ways with longtime distributor Warner Bros. Pictures, and signed with Universal Pictures for the biographical thriller Oppenheimer (2023), which won him Academy Awards for Best Director and Best Picture.

Nolan's work regularly features in the listings of best films of their respective decades. Infused with a metaphysical outlook, his films thematise epistemology, existentialism, ethics, the construction of time, and the malleable nature of memory and personal identity. They feature mathematically inspired images and concepts, unconventional narrative structures, practical special effects, experimental soundscapes, large-format film photography, and materialistic perspectives. His enthusiasm for the use and preservation of traditional film stock in cinema production as opposed to digital cameras has also garnered significant attention. He has co-written several of his films with his brother, Jonathan, and runs the production company Syncopy Inc. with his wife, Emma Thomas."""

    summary_template = """ Given the information {information} about a person, I want you to create:
    1. A short summary  
    2. Two interesting facts about them
    3. best movie they directed
    """

    summary_prompt_template = PromptTemplate(input_variables = ["information"], template= summary_template)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  
        google_api_key=os.getenv("GOOGLE_API_KEY"),        
        temperature=0
    )
    chain = summary_prompt_template | llm | StrOutputParser()
    result = chain.invoke(input={"information": information})
    print(result)
    
if __name__ == "__main__":
    main()

