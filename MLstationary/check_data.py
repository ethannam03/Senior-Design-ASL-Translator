import pandas as pd

df = pd.read_csv('training_data.csv')

print(f"Total samples: {len(df)}")
print(f"Number of unique letters: {df['label'].nunique()}")
print("\nSamples per letter:")
print(df['label'].value_counts().sort_index())

# Flag any letters with noticeably fewer samples than the rest
counts = df['label'].value_counts()
avg = counts.mean()
low_samples = counts[counts < avg * 0.5]  # less than half the average

if not low_samples.empty:
    print("\nThese letters have noticeably fewer samples than average:")
    print(low_samples)
else:
    print("\nSample counts look reasonably balanced.")