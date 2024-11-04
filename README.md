# HEALTHCARE-EDA
A BRIEF DATA ANALYSIS ON HEALTHCARE INFORMATION DATA
# Main Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Plots Adjustment
%matplotlib inline

# Handling Compiler Warnings
import warnings as ws
ws.filterwarnings('ignore')

# Code Insurance
print("Setup Complete")
# Setting Data Path
healthcare = "/content/healthcare_dataset.csv"

# Loading Data
df = pd.read_csv(healthcare)

# Display First 5 Rows
df.head()
# Display Data Information
df.info()
# Display Data Shape (Rows x Columns)
df.shape
# Display Data Description
df.describe()
# Display Number of Distinct Element of All DataFrames
df.nunique()
# Looking for duplicate values
df.duplicated().sum()
# Dropping the duplicate values
df.drop_duplicates(inplace = True)
df.duplicated().sum()
# Looking for null values
df.isnull().sum()
# **Value Counts On the Features**
# Value counts on Gender Column
df.Gender.value_counts()
# Value Counts on Blood Type
df['Blood Type'].value_counts()
# Value counts on Medical Condition
df['Medical Condition'].value_counts()
# Value counts on Medication
df.Medication.value_counts()
# Value counts on Test Results
df['Test Results'].value_counts()
# Value Counts On Insurance Provider
df['Insurance Provider'].value_counts()
# Value Count on Billing Amount
df['Billing Amount'].value_counts()
# Value Count on Room Number
df['Room Number'].value_counts()
# Value counts on Name
df['Name'].value_counts()
# **Visualizing the distribution of the features**
# Plotting: Age Distribution
plt.hist(df['Age'], bins=5, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show();
# Visualizing the distribution of the Features
object_columns = ['Gender', 'Blood Type', 'Medical Condition', 'Admission Type',
                  'Insurance Provider', "Medication", 'Test Results']

pastel_palette = sns.color_palette("pastel", len(object_columns))

for col in object_columns:
    plt.figure(figsize=(10, 6))  # Set figure size
    sns.countplot(data=df, x=col, palette=pastel_palette)

    # Add title and labels
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')

    # Show the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to fit labels
    plt.show();
# **Analysis On Hospital Visits**
# Count of hospital Visits by Gender
df['Gender'] = df['Gender'].astype('category')
sns.countplot(x='Gender', data=df)

plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Gender')
plt.show();
# Count of Hospital Visits By Blood Type
df['Blood Type'] = df['Blood Type'].astype('category')
sns.countplot(x='Blood Type', data=df)

plt.xlabel('Blood Type')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Blood Type')
plt.show();

# Count of Hospital Visits by Medical Condition
df['Medical Condition'] = df['Medical Condition'].astype('category')
sns.countplot(x='Medical Condition', data=df)

plt.xlabel('Medical Condition')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Medical Condition')
plt.show();
# Count of Hospital Visits by Insurance Provider
df['Insurance Provider'] = df['Insurance Provider'].astype('category')
sns.countplot(x='Insurance Provider', data=df)

plt.xlabel('Insurance Provider')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Insurance Provider')
plt.show();
# Count of Hospital Visits by Admission Type
df['Admission Type'] = df['Admission Type'].astype('category')
sns.countplot(x='Admission Type', data=df)

plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Admission Type')
plt.show();
# Count of Hospital Visits by Medication
df['Medication'] = df['Medication'].astype('category')
sns.countplot(x='Medication', data=df)

plt.xlabel('Medication')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Medication')
plt.show();
# Count of Hospital Visits by Test Results
df['Test Results'] = df['Test Results'].astype('category')
sns.countplot(x='Test Results', data=df)

plt.xlabel('Test Results')
plt.ylabel('Count')
plt.title('Count of Hospital Visits by Test Results')
plt.show();
# **Financial Aanalysis**
# Distribution of Billing Amounts
sns.boxplot(y='Billing Amount', data=df)
plt.title('Distribution of Billing Amounts')
plt.show()
plt.hist(df['Billing Amount'], bins=5, edgecolor='black')
plt.xlabel('Billing Amount')
plt.ylabel('Frequency')
plt.title('Histogram of Billing Amounts')
plt.show()
# a. Financial Analysis: Billing by Admission Type
# Boxplot: Billing Amount by Admission Type
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Admission Type', y='Billing Amount')
plt.title("Billing Amount by Admission Type")
plt.show()
# b. Financial Analysis: Billing by Admission Type
# Average Billing Amount by Insurance Provider
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Insurance Provider', y='Billing Amount')
plt.xticks(rotation=45)
plt.title("Average Billing Amount by Insurance Provider")
plt.show()
# **Analysis Based On Medical Condition**
# Average Age by Medical Condition
age_by_condition = df.groupby('Medical Condition')['Age'].mean().reset_index()

pastel_palette = sns.color_palette("pastel")
plt.figure(figsize=(10, 6))  # Adjust the figure size
sns.barplot(data=age_by_condition, x='Medical Condition', y='Age', palette=pastel_palette)

plt.title('Average Age by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Average Age')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Medication Distribution by Medical Condition
grouped_df = df.groupby(['Medical Condition', 'Medication']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Medical Condition', y='Count', hue='Medication', palette=pastel_palette)

plt.title('Medication Distribution by Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Count')

plt.xticks(rotation=45)
plt.legend(title='Medication')
plt.tight_layout()
plt.show()
# Patient Count by Gender and Medical Condition
sex_by_condition = df.groupby(['Medical Condition', 'Gender']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=sex_by_condition, x='Medical Condition', y='Count', hue='Gender', palette=pastel_palette)

plt.title('Patient Count by Gender and Medical Condition')
plt.xlabel('Medical Condition')
plt.ylabel('Patient Count')

plt.legend(title='Gender')

plt.xticks(rotation=45)
plt.tight_layout()

plt.show();
# Patient Count by Blood Type and Medical Condition
grouped_df = df.groupby(['Blood Type', 'Medical Condition']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Blood Type', y='Count', hue='Medical Condition', palette=pastel_palette)

plt.title('Patient Count by Blood Type and Medical Condition')
plt.xlabel('Blood Type')
plt.ylabel('Patient Count')

plt.legend(title='Medical Condition')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Patient Count by Admission Type and Medical Condition
grouped_df = df.groupby(['Admission Type', 'Medical Condition']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Admission Type', y='Count', hue='Medical Condition', palette=pastel_palette)

plt.title('Patient Count by Admission Type and Medical Condition')
plt.xlabel('Admission Type')
plt.ylabel('Patient Count')

plt.legend(title='Medical Condition')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
**Further Analysis On Patient Count, Blood Type, Gender, Admission Type, Test Results, Medication**
# Patient Count by Blood Type and Gender
grouped_df = df.groupby(['Blood Type', 'Gender']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))  # Adjust figure size
sns.barplot(data=grouped_df, x='Blood Type', y='Count', hue='Gender', palette=pastel_palette)

plt.title('Patient Count by Blood Type and Gender')
plt.xlabel('Blood Type')
plt.ylabel('Patient Count')

plt.legend(title='Gender')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Patient Count by Admission Type and Gender
grouped_df = df.groupby(['Admission Type', 'Gender']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Admission Type', y='Count', hue='Gender', palette=pastel_palette)

plt.title('Patient Count by Admission Type and Gender')
plt.xlabel('Admission Type')
plt.ylabel('Patient Count')

plt.legend(title='Gender')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Test Results Distribution by Admission Type
grouped_df = df.groupby(['Test Results', 'Admission Type']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Test Results', y='Count', hue='Admission Type', palette=pastel_palette)

plt.title('Test Results Distribution by Admission Type')
plt.xlabel('Test Results')
plt.ylabel('Count')

plt.legend(title='Admission Type')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Medication Distribution by Gender
grouped_df = df.groupby(['Medication', 'Gender']).size().reset_index(name='Count')

pastel_palette = sns.color_palette("pastel")

plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Medication', y='Count', hue='Gender', palette=pastel_palette)

plt.title('Medication Distribution by Gender')
plt.xlabel('Medication')
plt.ylabel('Count')

plt.legend(title='Gender')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Analysis On Admission Trends (Monthly)**
# Monthly Admission Trend
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
monthly_admissions = df['Date of Admission'].dt.month.value_counts().sort_index()

monthly_admissions_df = pd.DataFrame({'Month': monthly_admissions.index, 'Admissions': monthly_admissions.values})

monthly_admissions_df['Month'] = monthly_admissions_df['Month'].apply(lambda x:
    pd.to_datetime(str(x), format='%m').strftime('%B'))  # Convert month numbers to names

plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_admissions_df, x='Month', y='Admissions', marker='o')

plt.title('Monthly Admissions Trend')
plt.xlabel('Month')
plt.ylabel('Number of Admissions')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# **Analysis On  Average Length of Stay (ALOS)**
# a. Health Conditions and Admissions: Length of Stay by Admission Type and Medical Condition
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Calculating average Length of Stay by Admission Type
avg_los_by_admission = df.groupby('Admission Type')['Length of Stay'].mean().sort_values()

# Bar chart: Average Length of Stay by Admission Type
plt.figure(figsize=(10, 6))
avg_los_by_admission.plot(kind='bar', color='skyblue')
plt.title("Average Length of Stay by Admission Type")
plt.xlabel("Admission Type")
plt.ylabel("Average Length of Stay (days)")
plt.show()
# b. Calculate average Length of Stay by Medical Condition
avg_los_by_condition = df.groupby('Medical Condition')['Length of Stay'].mean().sort_values()

# Bar chart: Average Length of Stay by Medical Condition
plt.figure(figsize=(12, 6))
avg_los_by_condition.plot(kind='bar', color='salmon')
plt.title("Average Length of Stay by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Average Length of Stay (days)")
plt.xticks(rotation=45)
plt.show()
fig = px.sunburst(df, path = ["Test Results", "Medication"])
fig.update_traces(textinfo = "label + percent parent")
fig.update_layout(title_text = "Test Results by Medication",
                 titlefont = {'size' : 20, 'family' : 'Serif'},
                 width = 600, height = 600)
fig.show()
fig = px.sunburst(df, path = ["Admission Type", "Medical Condition"])
fig.update_traces(textinfo = "label + percent parent")
fig.update_layout(title_text = "Admission Type by Medical Condition",
                 titlefont = {'size' : 20, 'family' : 'Serif'},
                 height = 600, width = 600)
fig = px.sunburst(df, path = ["Gender", "Medical Condition", "Medication"])
fig.update_traces(textinfo = "label + percent parent")
fig.update_layout(title_text = "Patient's Status",
                 titlefont = {'size' : 20, 'family' : 'Serif'},
                 height = 600, width = 600)
fig.show()
# Most common Blood type
most_common_blood_type = df['Blood Type'].value_counts().idxmax()
print(f"The most common blood type among the patients is {most_common_blood_type}.")

# Number of Unique Hospitals
unique_hospitals = df['Hospital'].nunique()
print(f"There are {unique_hospitals} unique hospitals included in the dataset.")

# oldest patient in the dataset and his age
oldest_patient_age = df['Age'].max()
oldest_patient_name = df[df['Age'] == oldest_patient_age]['Name'].iloc[0]
print(f"The oldest patient in the dataset is {oldest_patient_name} with an age of {oldest_patient_age} years.")
# The doctor who treated the highest number of patient
doctor_highest_patient_count = df['Doctor'].value_counts().idxmax()
print(f"The doctor who has treated the highest number of patients is {doctor_highest_patient_count}.")
# Most common medication
most_frequent_medication = df['Medication'].value_counts().idxmax()
print(f"The most frequently prescribed medication is {most_frequent_medication}.")
# Average Billing amount per patient
average_billing_amount = df['Billing Amount'].mean()
print(f"The average billing amount for patients is ${average_billing_amount:.2f}.")
# Most Common Medical Condition (top 3)
top_three_medical_conditions = df['Medical Condition'].value_counts().head(3)
print("Top Three Most Common Medical Conditions:")
print("----------------------------------------")
print(top_three_medical_conditions)
# Insurance Provoder Analysis
# Number Of Patients & Overall Billing Amount Provided By Each Isurance Provider

# Aggregation & Grouping
provider = df.groupby(by = 'Insurance Provider').agg(
    {'Billing Amount': 'sum','Insurance Provider':'size'})

# Transformation
provider.columns = ['Billing Amount', 'Patients']
provider['Billing Amount'] = np.around(provider['Billing Amount'])

# Sorting
provider = pd.DataFrame(provider.sort_values(by ='Patients')).reset_index()

# Displaying
provider
