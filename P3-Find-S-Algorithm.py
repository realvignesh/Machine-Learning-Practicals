import csv

attributes = [['Sunny', 'Rainy'],
              ['Warm', 'Cold'],
              ['Normal', 'High'],
              ['Strong', 'Weak'],
              ['Warm', 'Cool'],
              ['Same', 'Change']]
num_attributes = len(attributes)

print("\nThe Most General Hypothesis: ['?', '?', '?', '?', '?', '?']")
print("The Most Specific Hypothesis: ['0', '0', '0', '0', '0', '0']")

a = []

print("\nThe Given Training Dataset \n")
with open("Datasets\\P3-Enjoy-Sport.csv") as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)
del a[0]

print("\nThe initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

print("\nFind S: Finding a Maximally Specific Hypothesis\n")
for i in range(0, len(a)):
    if a[i][num_attributes] == 'Yes':
        for j in range(0, num_attributes):
            if a[i][j] != hypothesis[j] and hypothesis[j] == '0':
                hypothesis[j] = a[i][j]
            elif a[i][j] != hypothesis[j] and hypothesis[j] != '0':
                hypothesis[j] = '?'
    print(f"For Training Example {i}, the hypothesis is {hypothesis}")

print("\nThe Maximally Specific Hypothesis for the given Training Examples:")
print(hypothesis)
