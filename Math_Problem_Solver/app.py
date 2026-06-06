import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StdOutCallbackHandler

# ----------------------------- #
# Streamlit Page Setup
# ----------------------------- #
st.set_page_config(page_title="Text to Math Problem Solver")
st.title("🧮 Text to Math Problem Solver Using Google Gemma2")

# ----------------------------- #
# API Key Input
# ----------------------------- #
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
    st.stop()

# ----------------------------- #
# Initialize LLM
# ----------------------------- #
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

# ----------------------------- #
# Wikipedia Tool
# ----------------------------- #
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Use this tool to look up topics or get more background information."
)

# ----------------------------- #
# Math Calculation Tool
# ----------------------------- #
math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Use this to perform precise mathematical calculations when given a math expression."
)

# ----------------------------- #
# Reasoning Prompt Template
# ----------------------------- #
prompt_template = """
You are an expert math problem solver.

Follow these steps carefully:
1. Understand and reason about the problem step-by-step.
2. Then write ONLY the *final mathematical expression* needed for the answer in the format below.

Question: {question}

Your output MUST be formatted exactly like this:

Reasoning: <your reasoning text>
Math Expression: <single-line mathematical expression to calculate>
"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

# ----------------------------- #
# Reasoning Tool
# ----------------------------- #
llm_chain = LLMChain(llm=llm, prompt=prompt)
reasoning_tool = Tool(
    name="Reasoning Chain",
    func=llm_chain.run,
    description="Use this tool to reason step-by-step before solving math problems."
)

# ----------------------------- #
# Agent Initialization
# ----------------------------- #
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# ----------------------------- #
# Chat History
# ----------------------------- #
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hello! I'm your Math Problem Solver. Ask me any math-related question!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----------------------------- #
# User Input
# ----------------------------- #
question = st.text_area(
    "Enter your math problem here:"
)

# ----------------------------- #
# Button to Solve
# ----------------------------- #
if st.button("🔍 Solve Problem"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        st_cb = StdOutCallbackHandler(st.container())

        try:
            # Step 1: Get the reasoning + math expression
            reasoning_output = llm_chain.run({"question": question})
            st.info(reasoning_output)

            # Step 2: Extract the math expression
            math_expression = ""
            if "Math Expression:" in reasoning_output:
                math_expression = reasoning_output.split("Math Expression:")[1].strip()

            if math_expression:
                # Step 3: Compute using calculator
                result = math_chain.run(math_expression)
                response = f"{reasoning_output}\n\n✅ Final Answer: {result}"
            else:
                response = reasoning_output

            # Display response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.success(response)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a math problem to solve.")
