import os

# Define the input and output folder paths
input_folder = "C:/Users/uluso/OneDrive/Masaüstü/EE475PROJECT/Apple_Yolo/train/labels"  # Replace with your input folder path
output_folder = "C:/Users/uluso/OneDrive/Masaüstü/EE475PROJECT/new_batches_withoutclass"  # Replace with your output folder path

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all .txt files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".txt"):  # Process only .txt files
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, file_name)

        # Read the content of the input file
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        # Add "1 " at the beginning of each line (for YOLO class)
        modified_lines = ["1 " + line[2:] for line in lines]

        # Write the modified content to the output file
        with open(output_file_path, "w") as file:
            file.writelines(modified_lines)

print(f"Processing complete! Modified files are saved in '{output_folder}'.")
