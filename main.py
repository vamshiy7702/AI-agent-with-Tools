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

from pathlib import Path
Note_FILE='jsonfiles/note.json'
file_path = Path(Note_FILE)
if file_path.stat().st_size == 0:
    with open(Note_FILE,'w') as f:
        json.dump({'Notes':[]},f)
        
Task_File='jsonfiles/task.json' 
file_path = Path(Task_File)    
if file_path.stat().st_size==0:
    with open(Task_File,'w') as f:
        json.dump({},f)

@tool
def weather_tool(query:str) -> str:
    """Your are a Weather Tool and get the WEATHER INFO
    Args:
        query (str): City Name
    Returns:
        str: string
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
def save_note(query:str) -> str:
    
    """
    Your are the Note making tool
    Args :
        query(str) : input string
    
    Returns :
        str : string format
    """
    
    with open(Note_FILE,'r') as file:
        json_file=json.load(file)

    for key,value in json_file.items():
        value=value.append(query)

    with open(Note_FILE,'w') as file:
        json.dump(json_file,file)
    return 'Note is Saved'
   
@tool
def Retrieve_note(query:str) -> str:
    """
      Your are the Retriever tool to get the saves info
    """
            
    with open(Note_FILE,'r') as file:
        json_file=json.load(file)

    for key,value in json_file.items():
        if len(value)==0:
            return 'Your Note is Empty'
        else:
            with open(Note_FILE,'r') as f:
                data=json.load(f)
            return data
@tool
def add_task(query:str) -> str:
    """
    Your are the task manager tool
    """
    index=1
    with open(Task_File,'r') as file:
        json_file=json.load(file)    
    data={}
    for key,value in json_file.items():
        key={}
        data[index]=query
        value=data
    index+=1  
    with open(Task_File,'w') as file:
        json.dump(json_file,file)
    return 'Task was added'

my_prompt="""  
     Your are the helpFull AI AGENT.
     1. IF THE GIVEN QUERY RELATED TO THE WEATHER THAN CALL weather_tool THAN EXECUTE WEATHER IN THE CITY ACCORDINGLY DONT HALUCINATE ANSWERING QUESTION.
     2. IF THE GIVEN QUERY IS PARAGRAPH ASKING SUMMARY THAN CALL text_summary TOOL AND GIVE SUMMARY OF THE PARAGRAPH DONT HALUCINATE ANSWERING QUESTIONS.
     3. IF USER WANTED TO add the details in json file than add it.BUT BELOW ARE THE RULES SHOULD MUST FOLLOW.
        Only add the important text in the user question dont add un-nessesary details to file   
           Ex: 
           user: remember I have meeting at 2 pm save the note
           Give the information only "meeting at 2 pm" 
           RESULT : NOTE IS SAVED 
     4. IF USER ASKING ABOUT INFO ABOUT NOTE OR SHOW MY NOTE THEN CALL RETRIEVE_NOTE tool.If data is AVAILABLE THAN SHOW IF empty THEN "your Note is empty"
     5. If THE USER WANTED TO ADD THE TASK TO THE JSON FILE THAN CALL add_task tool and ADD It to json file as dict object RESPONDE TO THE USER "Task added".
           
    """
prompt = ChatPromptTemplate.from_messages(
        [
            ("system", my_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{query}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
tools=[weather_tool,text_summary,save_note,Retrieve_note,add_task]

llm=ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model="moonshotai/kimi-k2-instruct-0905",
        )

agent=create_openai_tools_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

result=agent_executor.invoke({"query":
    """ What was the weather in berlin? """
})

print("result:- ", result['output'])