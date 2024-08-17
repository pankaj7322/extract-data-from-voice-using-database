from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained GraphCodeBERT model and tokenizer
model_name = "microsoft/graphcodebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def generate_sql_query(input_text):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Generate SQL query using the model
    outputs = model(**inputs)
    
    # Get the logits and convert them to probabilities
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    
    # Post-process to extract SQL query based on the highest probability
    predicted_index = torch.argmax(probabilities, dim=1).item()
    
    # Assuming the model predicts SQL templates; you would need to map the prediction to an actual SQL query
    sql_template = "SELECT * FROM employees WHERE join_date > '2020-01-01'"  # Example template
    
    return sql_template

# Example usage
input_text = "List all employees who joined after 2020"
sql_query = generate_sql_query(input_text)
print("Generated SQL Query:", sql_query)
