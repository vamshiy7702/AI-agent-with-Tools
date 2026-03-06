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
import re

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"


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
    try:
        return weather_data
    except:
        return "Temporary issue.Try Again later"

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
    with open(Task_File,'r') as file:
        json_file=json.load(file)
    index=len(json_file)+1
    json_file[index]={
        'TaskName':query,
        "Status":"Pending"
    } 
    with open(Task_File,'w') as file:
        json.dump(json_file,file,indent=4)
    return 'Task was added'

@tool
def view_task(query:str) -> str:
    """
    Your are the view task tool
    """
    with open(Task_File,'r') as file:
        json_read=json.load(file)
        if len(json_read)==0:
            return 'Your Tasks are Empty'
        else:
            return json_read
@tool
def mark_task(query:str) -> str:
    """
    Your are the mark_task tool
    """
    def check_data():
        with open(Task_File,'r') as file:
            json_read=json.load(file)
            return json_read
    file_data=check_data()
    if file_data:
        with open(Task_File,'r') as file:
            json_mark=json.load(file)
            matches = re.findall(r'-?\d*\.?\d+',query)
            my_key = [int(x) for x in matches]
            if len(my_key)==0:
                return 'Your tasks are empty'
        for i in my_key:
            with open(Task_File,'w') as file:
                json_mark[str(i)]['Status']="completed"
                json.dump(json_mark,file,indent=4)                     
        return 'Successfully Updated'
    else:
        return 'Your tasks are empty'   
            
my_prompt="""  
     Your are the helpFull AI AGENT.
     BELOW ARE THE STRICT RULES YOU SHOULD FOLLOW.
    1. ** If user asking mutiple QUESTIONS/QUERYS at a time, DONT BE HALUCINATE,
    Take step by step for answering the question and take step by step calling the tools and execute.
    2. If the given query related to the weather in the city than call "weather_tool" than execute weather in city accordingly.
    EX:What is the weather in Surat?
    output:Current temperature in Surat is 32°C with clear sky.
    3. If the given query is paragraph asking summary than call "text_summary" tool and give summary of the paragraph.
    4. If user wants to add something to note than call "save_note" tool.
      BUT BELOW ARE THE RULES SHOULD MUST FOLLOW.
       ** Only add the important text in the user question dont add un-nessesary details to file 
           Ex: 
           user: remember I have meeting at 2 pm save the note
           Give the information only "meeting at 2 pm" 
           RESULT : NOTE IS SAVED.
    5. If user asking info about note or show my note than call "retrieve_note" tool.If data is available than show if empty than reply "your note is empty".
    6. If user wants to add the task to the file than call "add_task" tool and add the task to file and respond "Task added".
    7. IF the User wanted to view or show the tasks than call "view_task" tool.Give the Result in the step by step 
       Ex: 1.TaskName - Status
    8. IF the User Want to update the status of the tasks or mark the status of the tasks than call "mark_task" tool than Update the Status.
    9. Dont be Halucinate for answering the questions.
    """
prompt = ChatPromptTemplate.from_messages(
        [
            ("system", my_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{query}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
tools=[weather_tool,text_summary,save_note,Retrieve_note,add_task,view_task,mark_task]

llm=ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model="openai/gpt-oss-20b",
        )

agent=create_openai_tools_agent(llm,tools,prompt)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

result=agent_executor.invoke({"query":
    """ update my tasks 5,6 and 8 is completed  """
})

print("result:- ", result['output'])