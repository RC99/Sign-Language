import os

# Specify the directory path
directory = '/Users/reetvikchatterjee/Desktop/Dataset/ThankYou'

# List all files in the directory
files = os.listdir(directory)

# Count the number of files
num_files = len(files)

print(num_files)