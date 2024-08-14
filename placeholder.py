import os
import random
import time
import string

output_dir = '/cnvrg/output/placeholder'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

def create_random_file():
    # Generate a random number between 0 and 100
    random_number = random.randint(0, 100)
    
    # Generate a random filename
    random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + '.txt'
    
    # Create the full file path
    file_path = os.path.join(output_dir, random_filename)
    
    # Write the random number to the file
    with open(file_path, 'w') as file:
        file.write(str(random_number))
    
    print(f'Created file: {file_path} with number: {random_number}')

# Loop to create a new file every 10 minutes
while True:
    create_random_file()
    time.sleep(600)  # Sleep for 600 seconds (10 minutes)


