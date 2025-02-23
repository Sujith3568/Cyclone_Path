import numpy as np
import pandas as pd

# Sample dataset of images with ID, timestamp, latitude, and longitude
data = {
    "id": ["25", "27", "28","30","30(1)","31","32","32(1)","33","33(1)","33(2)","34","34(1)","35","35(1)","35(2)","35(3)","36","36(1)","36(2)","36(3)","37","38","39","40","40(1)","40(2)","40(3)","41","42","42(3)",
           "43","43(2)","44","44(1)","44(2)","44(3)","45","45(1)","45(2)","45(3)","45(4)","46","46(1)","46(2)","46(3)","47","47(1)","47(2)","47(3)","47(4)","48",
           "48(1)","48(2)","48(3)","49","49(1)","49(2)","50","50(1)","50(2)","51","51(1)","52","52(1)","53","53(1)","53(2)","53(3)","54","55","55(2)","56","57","57(1)","57(2)","58","58(1)","59","59(1)","59(2)",
           "59(3)","60","60(1)","60(2)","61","61(1)","61(2)","62","63","63(1)","63(2)","64","64(1)","64(2)","65","65(2)","65(3)","67","67(1)","68","69","69(1)","70","73","74","74(1)","74(2)",
           "75","77","77(1)","81","81(1)","82","82(1)","83(1)","84","84(1)","85","85(1)","85(2)","86","86(1)","86(2)","87","94","98","99","101","102","106","111","112","115","118","119","128"],
    "latitude": [10.65, 13.76, 10.31, 12.96, 9.00, 15.25, 13.04, 22.70, 15.57, 8.10, 7.80, 16.33, 17.94, 13.99, 13.16, 10.99, 22.70, 14.68, 10.58, 6.97, 9.64, 12.87, 14.33, 14.93, 12.17, 16.70, 19.55, 15.30, 12.49, 7.25, 16.50,
                 12.57, 24.23, 12.70, 18.47, 13.41, 9.51, 14.37, 8.30, 14.79, 20.16, 15.21, 17.51, 12.22, 15.92, 15.95, 17.16, 16.70, 11.89, 7.64, 8.40, 17.57, 10.20, 15.68, 15.35, 
                 12.37, 14.58, 22.80, 11.85, 14.62, 22.23, 18.28, 18.01, 14.18, 18.31, 12.55, 11.55, 12.90, 16.30, 15.13, 15.19, 15.63, 7.73, 13.68, 11.72, 22.59, 8.07, 15.57, 8.90, 12.09, 13.96, 10.55, 15.23, 15.02, 18.51, 8.41, 12.77, 12.37, 14.6, 
                12.74, 17.10, 8.79, 15.50, 20.18, 14.51, 17.07, 12.30, 15.13, 23.83, 18.20, 20.36, 20.36, 17.25, 14.18, 10.80, 15.15, 12.74, 18.90, 15.80, 11.40, 14.60, 17.83, 16.45, 13.37, 14.80, 20.66, 13.16, 14.77, 19.80, 13.16, 13.88, 18.98, 18.98,
                15.80, 14.47, 14.10, 13.60, 9.05, 18.90, 20.91, 16.32, 21.12, 12.59, 12.32, 17.95, 17.90, 13.48],
    "longitude":[80.51, 64.49, 52.10, 84.83, 80.10, 83.75, 16.74, 68.15, 84.28, 53.10, 86.20, 81.33, 88.07, 64.77, 59.50, 82.07, 69.55, 47.64, 80.41, 56.51, 56.21, 85.25, 89.54, 89.54, 84.25, 88.32, 65.55, 90.00, 84.93, 77.51, 84.70,
                 85.24, 58.95, 90.94, 65.73, 86.68, 82.10, 84.69, 84.07, 80.92, 59.89, 71.28, 64.27, 86.90, 89.60, 85.75, 85.74, 90.60, 61.89, 57.71, 82.70, 70.87, 56.60, 82.11, 85.05,
                 88.30, 71.21, 64.75, 91.45, 70.83, 64.35, 62.80, 93.28, 57.70, 72.83, 60.38, 83.77, 72.60, 89.70, 53.75, 84.26, 92.81, 87.51, 84.59, 89.41, 88.90, 87.35, 84.35, 85.10, 91.86, 83.92, 72.98, 88.21, 83.57, 90.26, 73.62, 49.86, 84.56, 56.55,
                 55.85, 92.33, 88.03, 73.30, 89.60, 58.38, 86.91, 88.74, 53.75, 62.65, 88.00, 64.63, 64.63, 90.75, 59.02, 82.20, 72.90, 80.98, 88.50, 69.60, 55.67, 62.20, 70.11, 72.65, 86.83, 90.30, 69.02, 51.90, 68.57, 71.30, 51.90, 68.30, 61.94, 61.94,
                 86.30, 61.30, 59.10, 56.40, 71.51, 71.40, 91.09, 83.91, 68.38, 85.47, 85.91, 85.90, 86.20, 55.74],
} 


length_id = len(data["id"])
length_latitude = len(data["latitude"])
length_longitude = len(data["longitude"])

# Print lengths
print("Length of ID:", length_id)
print("Length of Latitude:", length_latitude)
print("Length of Longitude:", length_longitude)
# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Generate 10 additional latitude and longitude points
def generate_additional_points(lat, lon, num_points=10):
    # Random variations within a small range for demonstration
    lat_points = lat + np.random.uniform(-0.5, 0.5, num_points)
    lon_points = lon + np.random.uniform(-0.5, 0.5, num_points)
    return np.array([lat_points, lon_points]).T  # 2D array with shape (num_points, 2)

# Add a column for additional points
df["additional_points"] = df.apply(
    lambda row: generate_additional_points(row["latitude"], row["longitude"]).tolist(), axis=1
)

# Save the labeled data to a CSV file
df.to_csv("labeled_cyclone_data.csv", index=False)

print("Labeled dataset with additional points:")
print(df)
