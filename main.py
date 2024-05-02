import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_google_genai import GoogleGenerativeAI


# Get the Google API key from environment variables
api_key = os.environ.get("GOOGLE_API_KEY")

# Get the Serper API key from environment variables
serper_api_key = os.environ.get("SERPER_API_KEY")

# Initialize the SerperDevTool for searching
search_tool = SerperDevTool()

# Initialize the GoogleGenerativeAI LLM
llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)

# Initialize the research_agent
research_agent = Agent(
    role="Researcher",
    goal="Find and summarize the latest AI news",
    backstory="""You're a researcher at a large company. 
                You're responsible for analyzing data 
                and providing insights to the business.""",
    verbose=True,
    llm=llm,
)

# Initialize the task
task = Task(
    description="Find and summarize the latest AI news",
    expected_output="A bullet list summary of the top 5 most important AI news",
    agent=research_agent,
    tools=[search_tool],
)

# Initialize the crew
crew = Crew(agents=[research_agent], tasks=[task], verbose=2)

# Kickoff the crew to perform the task
result = crew.kickoff()

# Print the result
print(result)
