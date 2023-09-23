import re

# Function to find range in list of indices
def find_ranges(indices):
    ranges = []
    index = 0
    total = len(indices)
    
    while index < total:
        start = indices[index]
        end = start

        while index + 1 < total and indices[index + 1] - indices[index] == 1:
            end = indices[index + 1]
            index += 1

        ranges.append([start, end])
        index += 1

    return ranges

# Function to split range according to size
def split_ranges(ranges, size = 150):
    new_ranges = []
    
    # Loop through ranges
    for start, end in ranges:
        
        #Case: Greater than size
        if end - start > size:
            num_subranges = (end - start) // size
            for i in range(num_subranges):
                new_start = (start + i * size) + 1
                new_end = start + (i + 1) * size
                new_ranges.append([new_start, new_end])
                
            # Remainings after split but less than size
            if end % size != 0:
                new_ranges.append([start + num_subranges * size, end])
                
        else:
            new_ranges.append([start, end])

    return new_ranges

# Function to get the csv files of the selected subjects
def filter_by(subject:int, files:list):
    
    selected_files = []
    
    # Regex pattern
    regex_pattern = r"OCDetect_(\d+)_"

    for file in files:
        match = re.search(regex_pattern, file)
        if match and int(match.group(1)) == subject:
            selected_files.append(file)
    
    return selected_files

# Function to get all subjects ids from list of filenames
def get_subjects_ids(files:list):
    subjects = []
    
    if not isinstance(files, list):
        raise TypeError("Argument should be of type list.")
    
    if files.empty():
        raise ValueError("Argument cannot be empty.")
        
    # Regex pattern
    regex_pattern = r"OCDetect_(\d+)_"
    
    for file in files:
        match = re.search(regex_pattern, file)
        subjects.append(match.group(1))
    
    return subjects

# Function to group files by subject's id
def group_by_subjects(files:list):
    
    grouped_files = {}
    
    # Regex pattern
    regex_pattern = r"OCDetect_(\d+)_"

    for file in files:
        match = re.search(regex_pattern, file)
        if match: 
            sub_id = int(str(match.group(1)))
            # Empty list add to Subject
            if sub_id not in grouped_files:
                grouped_files[sub_id] = []
            # Append filename
            grouped_files[sub_id].append(file)
    
    # Return sorted lists values
    return sort_dict_lists(grouped_files)

# Function to count strings occurances in values of a dict.
def get_str_counts(val:dict):
    """
    Performs countings of strings occurances.
    
    Parameters:
       val (dict): A dictionary containg the list of strings as values. 
    Returns:
        dict: A dictionary containing the counts for each strings.
    """
    
    str_counts = {}

    for lst in val.values():
        for item in lst:
            # If empty list add
            if item not in str_counts:
                str_counts[item] = 0
            # Increase count
            str_counts[item] += 1
    
    return str_counts

# # Function to print dictionary
# def print_table(data, headers):
#     # Ensure headers are same
#     num_columns = len(headers)
    
#     # Convert dictionaries into lists
#     rows = [[i+1, k,data[k]] for i, k in enumerate(data)]
    
#     # Prepend "S.N." to the headers
#     headers_with_sn = ["S.N."] + headers

#     # Print the table using the tabulate library
#     print(tabulate(rows, headers=headers_with_sn, tablefmt="grid", numalign="right"))
    
    
# Function to sort values of dictionary
def sort_dict_lists(input_dict):
    """
    Sort the list values of each key in a dictionary.
    
    Args:
        input_dict (dict): Input dictionary containing keys and lists as values.
        
    Returns:
        dict: Dictionary with sorted list values for each key.
    """
    sorted_dict = {}
    
    for key, value in input_dict.items():
        sorted_dict[key] = sorted(value)
    
    return sorted_dict


# Split list into k-folds

def divide_into_groups(elements, k):
    if k <= 0:
        raise ValueError("Number of groups 'k' must be greater than 0")

    n = len(elements)
    group_size = n // k
    remainder = n % k

    groups = []
    start = 0

    for i in range(k):
        group_end = start + group_size + (1 if i < remainder else 0)
        groups.append(elements[start:group_end])
        start = group_end

    return groups
