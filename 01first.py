import os
from dotenv import load_dotenv

load_dotenv()

print("lETS CODE ....")
print(os.getenv("LANG_CHAIN_PROJECT"))

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANG_CHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true" #enalble llog trace

# chatGrok is like a driver thsat lets u talk to grok's llm
# from langchain_groq import ChatGroq;

# m = ChatGroq(
#     model="openai/gpt-oss-20b"
# )

# print(m)

# response = m.invoke("Who is Mayank Jajpure?")

# print(response)






# from langchain_groq import ChatGroq

# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.0,
#     max_retries=2,
#     # other params...
# )


# response = model.invoke("Who is Mayank Jajpure?")

# print(response)



from langchain_core.prompts import ChatPromptTemplate

# prompt= ChatPromptTemplate.from_messages(
#     [
#      ("system", "yu are an aI expert Agentic AI engenner. Provide the anser based on the question"), 
#      ("user","{input}")
#     ]
# )
# print(prompt)


prompt= ChatPromptTemplate.from_messages(
    [
     ("system", "you are a comedian/joker"), 
     ("user","{input}")
    ]
)
print(prompt)


from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=1
    )
print(model)



chain = prompt|model
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

print(chain)
response = chain.invoke({"input":"who are you ? and What is MAchine Learning, expalin in very short"})
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

print(response.content)