import pandas as pd

# Create a dataframe from a list of dictionaries
rectangles = [
    { 'height': 40, 'width': 10 },
    { 'height': 20, 'width': 9 },
    { 'height': 3.4, 'width': 4 }
]

rectangles_df = pd.DataFrame(rectangles)
#print(rectangles_df)

# Use the height and width to calculate the area
def calculate_area(row):
    row["area"] = row['height'] * row['width']
    row["perimeter"] = 2 * (row['height'] * row['width'])
    return row



obj = rectangles_df.apply(calculate_area, axis=1)
print(obj)