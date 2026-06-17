import snowflake.connector
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)


curr = conn.cursor()

sql = 'SELECT * FROM housing_staging'

df = curr.execute(sql).fetch_pandas_all()
# df['PRICE'] = df['PRICE'].astype(float)
# df['DATE'] = pd.to_datetime(df['DATE'])
# df['year_posted'] = df['DATE'].dt.year
# df['month_posted'] = df['DATE'].dt.month
# df['day_posted'] = df['DATE'].dt.day
# df = df.drop(columns=['DATE'])


# print(df.dtypes)





