from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sqlite3
import warnings
warnings.filterwarnings('ignore')

tokenizer = AutoTokenizer.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")

def get_prompt(tables, question):
    prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return prompt

def prepare_input(question: str, tables: dict[str, List[str]]):
    tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
    tables = ", ".join(tables)
    prompt = get_prompt(tables, question)
    input_ids = tokenizer(prompt, max_length=512, return_tensors="pt").input_ids
    return input_ids

def inference(question: str, tables: dict[str, List[str]]) -> str:
    input_data = prepare_input(question=question, tables=tables)
    input_data = input_data.to(model.device)
    outputs = model.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result

output = inference("show the name, salary and age of employee whole salary greater than 90000", {"employees": ["name","age","department","salary"]})

def execute_query_and_show_results(query:str):
    conn = sqlite3.connect('employee_database.db')
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        for row in results:
            print(row)
    except sqlite3.Error as e:
        print(f"An Error occured:{e}")
    conn.commit()
    conn.close()

execute_query_and_show_results(output)
