import speech_recognition as sr
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Say something!")
    audio = recognizer.listen(source)

# Convert the captured audio to text using Google Speech Recognition
try:
    text = recognizer.recognize_google(audio)
    print("You said: " + text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError:
    print("Could not request results from Google Speech Recognition service")

# Function to prepare the prompt for the model
def get_prompt(tables, question):
    prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return prompt

# Function to prepare the input for the model
def prepare_input(question: str, tables: dict[str, List[str]]):
    tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
    tables = ", ".join(tables)
    prompt = get_prompt(tables, question)
    
    # Explicitly enable truncation
    input_ids = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt").input_ids
    return input_ids

# Function to run inference using the model
def inference(question: str, tables: dict[str, List[str]]) -> str:
    input_data = prepare_input(question=question, tables=tables)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result

# Generate the SQL query from the spoken text
output = inference(text, {"employees": ["name", "age", "department", "salary"]})

# Function to execute the generated SQL query and display the results
def execute_query_and_show_results(query: str):
    conn = sqlite3.connect('employee_database.db')
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            print(row)
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    conn.commit()
    conn.close()

# Execute the SQL query and display the results
execute_query_and_show_results(output)
