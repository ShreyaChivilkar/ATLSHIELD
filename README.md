# 🛡️ ATLShield: Crime Forecasting and Patrol Recommendation System

ATLShield is a Flask-based API that uses machine learning models to:

- 🔍 Predict **hourly crime volumes** for a given neighborhood and month
- 🚓 Recommend the best **patrolling strategy** per hour
- 📊 Generate visualizations to help law enforcement with decision-making

All predictions are backed by trained **Random Forest models** using real Atlanta crime data.

---




## ⚙️ Setup Instructions 
### Python backend: Run on AWS! -- t2.small instance

1. **Clone the repo and enter the directory**  
   ```bash
   git clone <repo-url>
   cd atlShield

2. **Give access to Shell script**
   ```bash
   chmod +x run.sh

3. **Run Shell script to start Python service with nohup**
   ```bash
   ./run.sh

3. **Tail logs**
   ```bash
   tail -f atlShield.log


### 📊 Tableau Visualizations

All dashboards developed as part of this project are published on [Tableau Public](https://public.tableau.com/app/profile/shreya.chivilkar/viz/CrimeAnalysisAtlantaDashboard/Dashboard1?publish=yes).

You can also download the workbook from the same link to explore the design, layout, and implementation details.


--- 
## 📁 Project Structure

### Exploratory Data Analysis (EDA)

Before building the prediction models, we conducted a thorough EDA on Atlanta crime data from **2021 to 2025**. Key steps:

#### 🔍 Data Cleaning
- **Missing Value Handling**:  
  - Dropped columns with more than **30% missing values**.
  - Rows missing critical fields (e.g., `Zone`, `Beat`, `LocationType`, `ReportDate`, `Latitude`, `Longitude`) were removed to maintain data integrity.

- **Date Fixes**:  
  - Cleaned and standardized `ReportDate`.  
  - If `ReportDate` was invalid or missing, it was imputed using `OccurredToDate`.

#### Offense Mapping
- Categorized offenses into **Group A** and **Group B** using FBI’s **NIBRS offense codes**.
- Created an `offense_code_map` dictionary to easily filter and group crime types.

#### Why this matters?
These preprocessing steps ensured that:
- The model was trained on high-quality, consistent data.
- Time-based trends (like seasonal crime variations or hourly patterns) were accurately captured.
- Crime types could later be analyzed for patrolling strategy recommendations and clustering (hotspot analysis).

---
### Code Overview

- `atlShield.py` hosts a Flask backend with two endpoints:
  - `/predict_hourly_crime`: Predicts hourly crime volume (using a `RandomForestRegressor`) and recommends patrol strategies (using a `RandomForestClassifier`) based on historical data, time, and location.
  - `/average_daily_crime`: Returns the average number of crimes per day for a given neighborhood and month.
- Models are trained **at startup** using grouped, encoded features from `updated_patrol_types.csv`.
- Visual output: A crime-intensity bar chart is generated (with red/yellow/green bars) per request and saved to disk.
- Everything runs on a **headless backend** (`matplotlib.use('Agg')`) to ensure compatibility with servers like AWS EC2.
- **Hosted on AWS EC2 instance** for scalable deployment.
- **Integrated with Tableau (via locally running TabPy)** for interactive, real-time predictions directly from dashboards.


