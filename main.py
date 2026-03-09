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

from fastapi import FastAPI

app=FastAPI()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"


weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=os.getenv('OPENWEATHERMAP_API_KEY'))

from pathlib import Path
Note_FILE='jsonfiles/note.json'

Task_File='jsonfiles/task.json' 

        
def user_query(query:str) ->str:
    Note_FILE='jsonfiles/note.json'
    file_path = Path(Note_FILE)
    if file_path.stat().st_size == 0:
        with open(Note_FILE,'w') as f:
            json.dump({'Notes':[]},f)
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

        for value in json_file.values():
            value=value.append(query)

        with open(Note_FILE,'w') as file:
            json.dump(json_file,file)
        return 'Note is Saved'

    @tool
    def Retrieve_note() ->str:
        """
        Your are the Retriever tool to get the saves info
        """         
        with open(Note_FILE,'r') as file:
            json_file=json.load(file)
            
        for value in json_file.values():
            if len(value)==0:
                return 'Your Note is Empty'
            else:
                with open(Note_FILE,'r') as f:
                    data=json.load(f)
                return data
    @tool
    def add_task(query:str) -> str:
        """
        Your are the adding the tasks to file
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
    def view_task() ->str:
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
    def mark_task(number:int) -> str:
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
            index_list=[]
            for key in json_mark.keys():
                if key==str(number):
                    index_list.append(key)
                else:
                    pass
            if len(index_list)==0:
                return 'Your task is Not there'       
            with open(Task_File,'w') as file:
                json_mark[str(number)]['Status']="completed"
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
        3. If the given query is paragraph asking summary of that paragraph than give summary of the paragraph.
        4. If user wants to add something to note than call "save_note" tool.
        BUT BELOW ARE THE RULES SHOULD MUST FOLLOW.
        ** Only add the important text in the user question dont add un-nessesary details to file 
            Ex: 
            user: remember I have meeting at 2 pm save the note
            Give the information only "meeting at 2 pm" 
            RESULT : NOTE IS SAVED.
        5. If user asking info about note or show my note than call "retrieve_note" tool.If data is available than show if empty than reply "your note is empty".
        6. If user wants to add the tasks to the file than call "add_task" tool and add the tasks to file and responde.
        ** If user adding multiple tasks than take step by step each one add accordingly DONT BE HALUCINATE.
        7. IF the User wanted to view or show the tasks than call "view_task" tool.Give the Result in the step by step 
        Ex: 1.TaskName - Status
        8. IF the User Want to update the status of the tasks or mark the status of the tasks than call "mark_task" tool than Update the Status.
        Below are the rules follow are mark_task tool:
        ** If you find any INTEGERS value in users query than pass through tool and update the tasks in file as completed.
        Else If you find words in the query that are INTEGERS than take that INTEGER value and update the tasks in the file as completed
        ex:
        query : Update the 1 and 2nd tasks are completed :
        Than Take 1 and 2 are INTEGERS  
        ex 2:
        query : third and fourth tasks are completed:
        than take 3 and 4 ARE INTEGERS 
        EX 3:
        query : thirteenth task is completed: 
        than take 13 ARE INTEGER.
        CONTINUE THE CALLING THE TOOL UNITIL FIND NUMBER.
        9. Dont be Halucinate for answering the questions.
        """
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", my_prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human",f"{query}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
    tools=[weather_tool,save_note,Retrieve_note,add_task,view_task,mark_task]

    llm=ChatGroq(
            groq_api_key=os.environ['GROQ_API_KEY'],
            model="openai/gpt-oss-20b",
            )

    agent=create_openai_tools_agent(llm,tools,prompt)

    agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)

    result=agent_executor.invoke({f"{query}":f"{query}"})


    return result['output']

@app.post('/')
def Tool_query(query:str) ->str:
    return user_query(query)