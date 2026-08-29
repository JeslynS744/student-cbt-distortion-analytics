import pandas as pd
import numpy as np
import string
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(BASE_DIR, 'data', 'raw', 'original_student_gform.csv')
output_data_path = os.path.join(BASE_DIR, 'data', 'processed', 'original_preprocessed_student_mental_health_data.csv')

try:
    df = pd.read_csv(raw_data_path)
except FileNotFoundError:
    print(f"Error: The file '{raw_data_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV file: {e}")
    exit()

df.columns = df.columns.str.strip()

column_mapping = {
    'Timestamp': 'timestamp',
    'Age': 'age',
    'Gender': 'gender',
    'College/ Institution': 'College',
    'Course': 'Course',
    'Year of Study': 'year_of_study',
    'Feeling nervous, anxious, or on edge?': 'feeling_nervous_anxious',
    'I have trouble sleeping due to anxious thoughts.': 'trouble_sleeping_anxious_thoughts',
    'Worrying too much about various things?': 'worrying_too_much',
    'I feel tired even after sleeping well.': 'feeling_tired_after_sleep',
    'I feel sad or empty most of the time.': 'feeling_sad_empty',
    'I find it hard to enjoy things I used to like.': 'difficulty_enjoying_things',
    'Are you aware of the concept of Cognitive Behavioral Therapy (CBT) ?': 'aware_of_cbt',
    'If yes, can you briefly describe what you understand about CBT?': 'cbt_description',
    'Do you experience any symptoms of anxiety or depression? If yes, please describe your experiences in a few sentences.': 'anxiety_depression_description'
}

df.rename(columns=column_mapping, inplace=True)

def standardize_text(text):
    if isinstance(text, str):
        text = text.lower().strip()
        text = text.replace("ocassionally", "occasionally")
        return text
    return text

for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].apply(standardize_text)

for col in df.columns:
    if df[col].dtype == 'object':
        mode = df[col].mode()
        fill_value = mode[0] if not mode.empty else 'Missing'
        df[col] = df[col].fillna(fill_value)
    else:
        if df[col].notna().sum() > 0:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            df[col] = df[col].fillna(0)

unified_mapping = {
    'never': 0, 'occasionally': 1, 'sometimes': 2, 'often': 3, 'always': 4,
    'strongly disagree': 0, 'disagree': 1, 'neutral': 2, 'agree': 3, 'strongly agree': 4
}
yes_no_mapping = {'no': 0, 'maybe': 1, 'yes': 2}

for col in df.columns:
    df[col] = df[col].map(unified_mapping).fillna(df[col])
    df[col] = df[col].map(yes_no_mapping).fillna(df[col])

def clean_text(text):
    if isinstance(text, str):
        return ''.join(char for char in text if char not in string.punctuation)
    return 'Missing'

for col in ['College', 'Course', 'cbt_description', 'anxiety_depression_description']:
    if col in df.columns:
        df[col] = df[col].apply(clean_text)

def extract_year(year_string):
    try:
        return int(''.join(filter(str.isdigit, str(year_string))))
    except:
        return np.nan

if 'year_of_study' in df.columns:
    df['year'] = df['year_of_study'].apply(extract_year)
    df.drop('year_of_study', axis=1, inplace=True)

def process_age(value):
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value)
        numbers = list(map(int, numbers))
        if len(numbers) == 2:
            return sum(numbers) / len(numbers)
        elif len(numbers) == 1:
            return numbers[0]
    elif isinstance(value, (int, float)):
        return value
    return np.nan

df['age'] = df['age'].apply(process_age)

try:
    df['age'] = df['age'].astype(float)
except ValueError:
    df['age'] = df['age'].apply(lambda x: np.nan if not str(x).isdigit() else float(x))

os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
df.to_csv(output_data_path, index=False)
print(f"Preprocessed data saved successfully to: {output_data_path}")