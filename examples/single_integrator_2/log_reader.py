import re

# Initialize lists to store violations for each run
actual_run_violations = []
measured_run_violations = []

# Variables to store the last occurrence of maximum violations
max_actual_violation = None
max_measured_violation = None

max_actual_violation_index = -1
max_measured_violation_index = -1

max_actual_count = -1
max_measured_count = -1
max_actual_count_index = -1
max_measured_count_index = -1

# Open and read the text file
with open("data/log_101_205", "r") as file:
    for index, line in enumerate(file):
        # Match actual run violations
        actual_match = re.search(r"Actual run violations:\s+(\d+)", line)
        if actual_match:
            actual_count = int(actual_match.group(1))
            actual_run_violations.append(actual_count)
            
            # Update max actual count and its index
            if actual_count > max_actual_count:
                max_actual_count = actual_count
                max_actual_count_index = index

        # Match measured run violations
        measured_match = re.search(r"Measured run violations:\s+(\d+)", line)
        if measured_match:
            measured_count = int(measured_match.group(1))
            measured_run_violations.append(measured_count)
            
            # Update max measured count and its index
            if measured_count > max_measured_count:
                max_measured_count = measured_count
                max_measured_count_index = index

        # Match max actual violation (take the last occurrence)
        max_actual_match = re.search(r"Max actual violation:\s+([\d.]+)", line)
        if max_actual_match:
            actual_violation = float(max_actual_match.group(1))
            max_actual_violation = actual_violation
            max_actual_violation_index = index  # Track the index

        # Match max measured violation (take the last occurrence)
        max_measured_match = re.search(r"Max measured violation:\s+([\d.]+)", line)
        if max_measured_match:
            measured_violation = float(max_measured_match.group(1))
            max_measured_violation = measured_violation
            max_measured_violation_index = index  # Track the index


print("Actual Violation Count Mean:", sum(actual_run_violations) / len(actual_run_violations) if actual_run_violations else 0)
print("Measured Violation Count Mean:", sum(measured_run_violations) / len(measured_run_violations) if measured_run_violations else 0)

# Divide each value by 20001 (total time steps)
actual_run_violations = [value / 20001 for value in actual_run_violations]
measured_run_violations = [value / 20001 for value in measured_run_violations]

# Calculate the average of divided values
average_actual = sum(actual_run_violations) / len(actual_run_violations) if actual_run_violations else 0
average_measured = sum(measured_run_violations) / len(measured_run_violations) if measured_run_violations else 0

print("Actual Violations Average Ratio:", average_actual)
print("Measured Violations Average Ratio:", average_measured)

print("Max Actual Violation:", max_actual_violation, "at indices", max_actual_violation_index)
print("Max Measured Violation:", max_measured_violation, "at index", max_measured_violation_index)