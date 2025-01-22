import pandas as pd
import re
import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PATH = "Reader"
FILE_TO_ANALYZE = "finetune"  
answer_column = 'Embedding Chunks'  # This should be the name of the column containing the answers
USE_COL = 'D'  # This should be the column to use for the analysis
USE_ROW = 7  # This should be the row to start reading from

def parse_answer(text):
    if pd.isna(text):
        return None, set()
    answer_match = re.search(r'Answer:\s*(Yes|No)', str(text), re.IGNORECASE)
    section_match = re.search(r'Section:\s*¬?ß?\s*([\d., ]+)', str(text))
    
    answer = answer_match.group(1) if answer_match else None
    sections = set(section.strip() for section in section_match.group(1).split(',')) if section_match else set()
    
    return answer, sections

def analyze_answer(given, truth):
    # Parse the given answer
    given_answer, given_sections = parse_answer(given)
    print("Given answer: ", given_answer, "Given sections: ", given_sections)
    # Parse the truth
    truth_parts = truth.split(',')
    truth_answer = truth_parts[0].strip()
    if truth_answer not in ['Yes', 'No']:
        truth_answer = 'Yes'  # Assume 'Yes' if only section is given
    truth_sections = set(section.strip() for section in truth_parts[1:] or truth_parts[0:1] if section.strip() not in ['Yes', 'No'])
    
    if not given_answer:
        return False, False, "Invalid format: Unable to parse answer"

    # Check if the answer is correct
    is_answer_correct = given_answer.lower() == truth_answer.lower()

    # Check sections
    if truth_sections:
        is_section_correct = any(section in truth_sections for section in given_sections)
    else:
        is_section_correct = True  # Consider section correct if not applicable in truth

    comment = f"Answer: {'Correct' if is_answer_correct else 'Incorrect'}"
    if truth_sections:
        comment += f", Section: {'Correct' if is_section_correct else 'Incorrect'}"

    return is_answer_correct, is_section_correct, comment

def process_excel(file_path, answer_column):
    # # Read the Excel file
    # df = pd.read_excel(file_path)
    
    # Read the Excel file, start reading from D8 for the Answer column
    df = pd.read_excel(file_path, usecols=USE_COL, skiprows=USE_ROW)

    # Define the truth table
    # Define the truth table with comments indicating the question index
    truth_table = [
        "Yes, 15.247",  # Question 1
        "Yes, 15.247",  # Question 2
        "No, 15.247",   # Question 3
        "No, 15.247",   # Question 4
        "No, 15.247",   # Question 5
        "Yes, 15.247",  # Question 6
        "No",           # Question 7
        "No",           # Question 8
        "No, 15.247, 15.407",   # Question 9
        "No, 15.247, 15.249",   # Question 10
        "No, 15.247, 15.249",   # Question 11
        "Yes",          # Question 12
        "No",           # Question 13
        "Yes, 15.247",  # Question 14
        "Yes, 15.247",  # Question 15
        "Yes, 15.247",  # Question 16
        "Yes",          # Question 17
        "Yes, 15.247",  # Question 18
        "Yes, 15.247",  # Question 19
        "Yes, 15.247",  # Question 20
        "Yes, 15.247",  # Question 21
        "No, 15.247",   # Question 22
        "Yes, 15.247",  # Question 23
        "No",           # Question 24
        "No, 15.247",  # Question 25
        "Yes, 15.247",  # Question 26
        "No, 15.407",   # Question 27
        "Yes",           # Question 28
        "No, 15.519",   # Question 29
        "No, 15.519",   # Question 30
        "Yes",          # Question 31
        "No",          # Question 32
        "Yes, 15.519, 15.517",  # Question 33
        "Yes",           # Question 34
        "Yes",           # Question 35
        "No",           # Question 36
        "Yes, 15.519",  # Question 37
        "No, 15.517, 15.519",  # Question 38
        "No, 15.517, 15.519",  # Question 39
        "No, 15.511, 15.517, 15.519", # Question 40
        "No, 15.515, 15.519",  # Question 41
        "No, 15.515, 15.519",  # Question 42
        "No, 15.515, 15.519",  # Question 43
        "No, 15.515",   # Question 44
        "Yes, 15.513",  # Question 45
        "Yes, 15.513",  # Question 46
        "Yes, 15.513",  # Question 47
        "No",           # Question 48
        "No",          # Question 49
        "Yes",          # Question 50
        "Yes",          # Question 51
        "No",          # Question 52
        "Yes, 15.247",          # Question 53
        "No, 15.247",           # Question 54
        "Yes, 15.247",  # Question 55
        "No"           # Question 56
    ]

    # Add the truth table to the DataFrame, matching the length of the existing data
    df['Truth'] = truth_table[:len(df)]

    # Apply the analysis to the specified columns
    df['Analysis'] = df.apply(lambda row: analyze_answer(row[answer_column], row['Truth']), axis=1)

    # Split the analysis results into separate columns
    df['Answer_Correct'] = df['Analysis'].apply(lambda x: x[0] if isinstance(x, tuple) else False)
    df['Section_Correct'] = df['Analysis'].apply(lambda x: x[1] if isinstance(x, tuple) else False)
    df['Comments'] = df['Analysis'].apply(lambda x: x[2] if isinstance(x, tuple) else "Error in analysis")

    # Drop the temporary 'Analysis' column
    df = df.drop('Analysis', axis=1)

    # Calculate statistics
    total_entries = len(df)
    incorrect_entries_yesno_incorrect = df[~df['Answer_Correct']]
    incorrect_entries_both_incorrect = df[~(df['Answer_Correct'] & df['Section_Correct'])]
    num_incorrect_yesno = len(incorrect_entries_yesno_incorrect)
    num_incorrect_both = len(incorrect_entries_both_incorrect)
    accuracy_percentage_yesno = ((total_entries - num_incorrect_yesno) / total_entries) * 100
    accuracy_percentage_both = ((total_entries - num_incorrect_both) / total_entries) * 100

    # Prepare the output
    output = f"Total entries: {total_entries}\n"
    output += f"Number of incorrect Yes/No entries: {num_incorrect_yesno}\n"
    output += f"Number of incorrect Yes/No and Section entries: {num_incorrect_both}\n"
    output += f"Accuracy for Yes/No: {accuracy_percentage_yesno:.2f}%\n"
    output += f"Accuracy for Yes/No and Section: {accuracy_percentage_both:.2f}%\n\n"
    output += "Incorrect entries:\n"
    
    for index, row in incorrect_entries_both_incorrect.iterrows():
        output += f"Row {index + 1}: \n"  # +1 because Excel rows start at 1
        output += f"Given answer: {row[answer_column][:100]}...\n"  # Truncate for readability
        output += f"Truth: {row['Truth']}\n"
        output += f"Analysis: {row['Comments']}\n\n"

    # Ensure the output directory exists
    output_dir = os.path.join(SCRIPT_DIR, FOLDER_PATH, FILE_TO_ANALYZE)
    os.makedirs(output_dir, exist_ok=True)

    # Save the results back to Excel
    output_file = os.path.join(output_dir, f'[Analyzed]_{FILE_TO_ANALYZE}.xlsx')
    df.to_excel(output_file, index=False)

    print(f"Analysis complete. Results saved to {output_file}")
    print("\nStatistics:")
    print(output)

    # Also save the statistics to a text file
    stats_file = os.path.join(output_dir, f'[Statistics]_{FILE_TO_ANALYZE}.txt')
    with open(stats_file, 'w') as f:
        f.write(output)
    print(f"Detailed statistics saved to {stats_file}")

if __name__ == "__main__":
    # Process the Excel file
    file_path = os.path.join(SCRIPT_DIR, FOLDER_PATH, f'{FILE_TO_ANALYZE}.xlsx')  # Update this to your actual file name
    process_excel(file_path, answer_column)

    print("\nExplanation of analysis:")
    print("""
    This script analyzes the 'Answer' column of your existing Excel file against a predefined truth table.
    It extracts the Yes/No answer and section number(s) from each answer and compares them to the truth table.
    The results show which answers are correct in terms of both the Yes/No response and the section numbers.
    Incorrect entries are listed in detail in the output.
    """)