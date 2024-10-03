import pandas as pd

# Load the dataset
df = pd.read_csv('Unemployment.csv')

# Check the columns and the first few rows
print("Column Names:")
print(df.columns)
print("\nFirst Few Rows:")
print(df.head())

# If 'Date' is not found, check for leading/trailing spaces
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces

# Try accessing the 'Date' column again
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("Column 'Date' not found. Available columns are:")
    print(df.columns)

# If 'Date' is available, proceed with your analysis
if 'Date' in df.columns:
    df.set_index('Date', inplace=True)

    # Plotting and further analysis here...
    plt.figure(figsize=(14, 7))
    plt.plot(df['Estimated Unemployment Rate (%)'], marker='o', linestyle='-')
    plt.title('Unemployment Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.grid()
    plt.show()
else:
    print("Please check your dataset for the correct date column.")

