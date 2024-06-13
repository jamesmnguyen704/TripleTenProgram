# TripleTen
Projects and Task
libraries
pandas
plotly.express
numpy
seaborn
matplotlib.pyplot
Exploratory Data Analysis of Vehicle Sales Data
Introduction
This project focuses on conducting an exploratory data analysis (EDA) of a dataset containing vehicle sales data. The primary objectives are to preprocess the data, address missing values and outliers, and uncover trends and insights through visualizations.

Dataset
The dataset, vehicles_us.csv, includes various attributes of vehicles listed for sale, such as price, model year, model, condition, cylinders, fuel type, odometer reading, transmission type, vehicle type, paint color, four-wheel-drive availability, date posted, and days listed.

Methodology
Data Preprocessing
Filling Missing Values:

model_year: Filled missing values by grouping by model and using the median year.
cylinders: Missing values filled by median cylinders for each model group.
odometer: Missing values filled by median or mean odometer reading, grouped by model year or model year plus model.
Outlier Removal:

Removed outliers in model_year and price to enhance the quality of scatterplots and other visualizations.
Data Visualization
Several types of visualizations were used to explore the data:

Histograms for distribution analysis.
Scatter plots to understand relationships between variables like price and model year.
Bar charts to compare average prices across different categories.
Running the Project
Jupyter Notebook (EDA.ipynb)
To view the exploratory data analysis, follow these steps:

Ensure that Python, Jupyter Notebook, and required libraries (Pandas, Matplotlib, Seaborn, Plotly, etc.) are installed.
Open EDA.ipynb in Jupyter Notebook.
Run each cell to see the analysis and visualizations.
Streamlit App (app.py)
To run the Streamlit web application:

Ensure Streamlit is installed (pip install streamlit).
Run the app using streamlit run app.py in your terminal.
View the app in your web browser as prompted.

Graphs
Scatterplot
![Number of Distribution](image-1.png)
