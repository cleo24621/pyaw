import numpy as np

# Original datetime string
datetime_string = "20160311T064700"

# Insert delimiters to make it ISO 8601 compliant
formatted_string = datetime_string[:4] + "-" + datetime_string[4:6] + "-" + datetime_string[6:8] + "T" + datetime_string[9:11] + ":" + datetime_string[11:13] + ":" + datetime_string[13:]

# Convert to numpy.datetime64 with nanosecond precision
datetime_obj = np.datetime64(formatted_string, 'ns')

print(datetime_obj)
