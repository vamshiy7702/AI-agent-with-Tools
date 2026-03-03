from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.tools import tool

from langchain_classic.agents import AgentExecutor
from langchain_classic.agents import create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
import json

weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=os.getenv('OPENWEATHERMAP_API_KEY'))

@tool
def weather_tool(query:str) -> str:
    """Your are a Weather Tool and get the WEATHER INFO
    Args:
        query (str): City Name

    Returns:
        str: Structure string format only give Current Temperature 
    """
    weather_data = weather.run(query)
    
    return weather_data

@tool
def text_summary(query:str) -> str:
    """ 
    Text summary tool 
    Args : 
        query (str): Paragraph
    Returns:
        str : Short Summary Generate 
    """
    return query

@tool
def note_tool(query:str) -> str:
    """
    Your are the Note making tool 
    
    Args :
        query(str) : input string
    
    Returns :
        str : string format
    """
    try:
        with open("jsonfiles/note.json", "r") as f:
            data = json.load(f)
        return data
    except:
        with open('jsonfiles/note.json', 'a') as outfile:
            outfile.write(json.dumps(query))
            outfile.write(",")
            outfile.close()
            return 'Note saved'


my_prompt="""  
     Your are the helpFull AI AGENT.
     1. IF THE GIVEN QUERY RELATED TO THE WEATHER THAN CALL weather_tool THAN EXECUTE WEATHER IN THE CITY ACCORDINGLY.
     2. IF THE GIVEN QUERY IS RELATED TO THE PARAGRAPH THAN CALL text_summary TOOL AND GIVE SUMMARY OF THE PARAGRAPH.
     3. IF USER WANTED TO get the information of notes details than get it and BREAK ELSE SAVE THE NOTE.
    """

prompt = ChatPromptTemplate.from_messages(
        [
            ("system", my_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{query}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

tools=[weather_tool,text_summary,note_tool]

llm=ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model="llama-3.3-70b-versatile",
        )

agent=create_openai_tools_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

result=agent_executor.invoke({"query":
    """save the note : schedule the meeting at 8 pm """
})

print(result['output'])