import torch
import pyaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import time
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")
model1 = AutoModelForSeq2SeqLM.from_pretrained("juierror/flan-t5-text2sql-with-schema-v2")


# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream for recording
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

# Record for 10 seconds
start_time = time.time()
while time.time() - start_time < RECORD_SECONDS:
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording stopped.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Convert the frames into a numpy array
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# Normalize the audio data
audio_data = audio_data / np.max(np.abs(audio_data), axis=0)

# Process the audio data with the Whisper processor
inputs = processor(audio_data, sampling_rate=RATE, return_tensors="pt", language="en")

# Manually create the attention mask
attention_mask = torch.ones(inputs.input_features.shape, dtype=torch.long)

# Move input data and attention mask to the correct device
input_features = inputs.input_features.to(device)
attention_mask = attention_mask.to(device)

# Generate token ids using the input features and attention mask
predicted_ids = model.generate(input_features, attention_mask=attention_mask)

# Decode token ids to English text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, language="en")[0]

print("Transcription:", transcription)


# Function to prepare the prompt for the model
def get_prompt(tables, question):
    prompt = f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return prompt

# Function to prepare the input for the model
def prepare_input(question: str, tables: dict[str, list[str]]):
    tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
    tables = ", ".join(tables)
    prompt = get_prompt(tables, question)
    
    # Explicitly enable truncation
    input_ids = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt").input_ids
    return input_ids

# Function to run inference using the model
def inference(question: str, tables: dict[str, list[str]]) -> str:
    input_data = prepare_input(question=question, tables=tables)
    input_data = input_data.to(model1.device)
    outputs = model1.generate(inputs=input_data, num_beams=10, top_k=10, max_length=512)
    result = tokenizer.decode(token_ids=outputs[0], skip_special_tokens=True)
    return result

# Generate the SQL query from the spoken text
output = inference(transcription, {"employees": ["name", "age", "department", "salary"]})

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
