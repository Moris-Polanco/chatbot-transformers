import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer

# Setting up the GPT-3 model
model_name = "openai/gpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

# App title
st.title("GPT-3 Chatbot")

# Creating the chatbot
user_input = st.text_input("User:")

if user_input != "":
    # Tokenizing the user's input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generating a response to the user's input
    response_ids = model.generate(
        input_ids, max_length=50, do_sample=True, top_k=50
    )
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Displaying the response
    st.write("GPT-3:", response)
