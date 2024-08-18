import sqlite3
import random
import faker

# Initialize Faker to generate random names and other data
fake = faker.Faker()

# Connect to the database (it will be created if it doesn't exist)
conn = sqlite3.connect('employee_database.db')

# Create a cursor object
cursor = conn.cursor()

# Create the employees table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        department TEXT,
        salary REAL
    )
''')

# List of departments to choose from
departments = ['HR', 'Finance', 'IT', 'Marketing', 'Sales', 'Support']

# Insert 100 random records into the employees table
for _ in range(100):
    name = fake.name()
    age = random.randint(22, 60)
    department = random.choice(departments)
    salary = round(random.uniform(50000, 120000), 2)
    
    cursor.execute('''
        INSERT INTO employees (name, age, department, salary)
        VALUES (?, ?, ?, ?)
    ''', (name, age, department, salary))

# Commit the changes
conn.commit()

# (Optional) Fetch and display the data to verify
cursor.execute("SELECT * FROM employees")
rows = cursor.fetchall()

print("Employees:")
for row in rows:
    print(row)

# Close the connection
cursor.close()
conn.close()
