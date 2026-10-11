# ============================================================================
# PURPOSE: Demonstrates a complete LangChain pipeline with Groq's LLaMA model
# This script loads environment variables, sets up API credentials, creates 
# a prompt template, initializes a language model, and invokes it with output parsing
# ============================================================================

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This securely reads API keys and project configurations without hardcoding them
load_dotenv()

print("lETS CODE ....")
print(os.getenv("LANG_CHAIN_PROJECT"))

# ============================================================================
# Set up API credentials in environment variables
# These are required for authenticating with external services
# ============================================================================
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Groq API key for LLM access
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # LangChain API key for tracking
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANG_CHAIN_PROJECT")  # LangChain project identifier
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable logging and tracing for debugging

# ============================================================================
# SECTION 2: IMPORTS - LangChain Components
# ============================================================================
print("Output pareser")

# StrOutputParser: Converts the model's response (ChatMessage object) into a plain string
from langchain_core.output_parsers import StrOutputParser

# ChatPromptTemplate: Combines multiple message templates into a structured prompt
# SystemMessagePromptTemplate: Provides system-level instructions (model behavior, tone)
# HumanMessagePromptTemplate: Template for user input with placeholders
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  

# ChatGroq: LangChain wrapper for Groq's API (fast, efficient LLM inference)
from langchain_groq import ChatGroq

# ============================================================================
# SECTION 3: DEFINE PROMPT TEMPLATES
# ============================================================================
# System message: Sets the model's personality and instructions
sys_msg = SystemMessagePromptTemplate.from_template("You are a consise assistant. give short factual answers.")

# User message: Creates a placeholder {question_text} for dynamic input
user_msg = HumanMessagePromptTemplate.from_template("{question_text}")

# ============================================================================
# SECTION 4: CREATE THE CHAT PROMPT TEMPLATE
# ============================================================================
# Combines system and user messages into a complete prompt structure
# The order matters: system message first, then user message
chat_template = ChatPromptTemplate.from_messages([sys_msg, user_msg])

# ============================================================================
# SECTION 5: INITIALIZE THE LANGUAGE MODEL
# ============================================================================
# Creates a ChatGroq instance with specific parameters:
# - model: "openai/gpt-oss-20b" - The specific model to use
# - temperature: 0.0 - Deterministic responses (no randomness, higher = more creative)
# - max_retries: 2 - Retry failed requests up to 2 times
groq_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0, max_retries=2)

# ============================================================================
# SECTION 6: INITIALIZE THE OUTPUT PARSER
# ============================================================================
# StrOutputParser extracts plain text from the model's ChatMessage response
# Removes metadata and returns only the answer string
text_only_parser = StrOutputParser()

# ============================================================================
# SECTION 7: BUILD THE PIPELINE (CHAIN)
# ============================================================================
# Uses the pipe operator (|) to chain components together:
# Prompt Template → Language Model → Output Parser
# This creates a data flow: question → formatted prompt → model response → parsed string
string_pipeline = chat_template | groq_model | text_only_parser

# ============================================================================
# SECTION 8: INVOKE THE PIPELINE
# ============================================================================
# Pass the question through the entire pipeline and get the result
# The {question_text} placeholder is replaced with "What is Generative AI?"
result = string_pipeline.invoke({"question_text": "What is Generative AI?"})

# ============================================================================
# SECTION 9: OUTPUT THE RESULT
# ============================================================================
# Print the final parsed result (plain string, not a ChatMessage object)
print("RAW parser output: ", result)



