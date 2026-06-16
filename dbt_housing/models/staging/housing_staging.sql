WITH raw_data AS (
    SELECT * FROM {{ source('housing_raw_data', 'housing_raw') }}
),

-- 1. Calculate global statistics once
global_stats AS (
    SELECT 
        MODE(TO_DATE(LEFT(date, 8), 'YYYYMMDD')) AS mode_date,
        MODE(bedrooms::INT) AS mode_bedrooms,
        MODE(sqft_living::INT) AS mode_sqft_living,
        MODE(sqft_lot::INT) AS mode_sqft_lot,
        MODE(ROUND(floors, 1)::INT) AS mode_floors,
        MEDIAN(view::INT) AS med_view,
        MEDIAN(sqft_above::INT) AS med_sqft_above,
        MEDIAN(sqft_basement::INT) AS med_sqft_basement,
        MEDIAN(yr_built::INT) AS med_yr_built,
        MEDIAN(zipcode::INT) AS med_zipcode,
        
        -- New statistics for the remaining columns
        MEDIAN(price::NUMERIC) AS med_price,
        MEDIAN(lat::FLOAT) AS med_lat,
        MEDIAN(long::FLOAT) AS med_long,
        MEDIAN(sqft_living15::INT) AS med_sqft_living15,
        MEDIAN(sqft_lot15::INT) AS med_sqft_lot15
    FROM raw_data
),

-- 2. Apply the imputations and cleaning rules
cleaned AS (
    SELECT 
        -- Original Columns
        COALESCE(TO_DATE(LEFT(r.date, 8), 'YYYYMMDD'), s.mode_date) AS date,
        COALESCE(r.bedrooms::INT, s.mode_bedrooms) AS bedrooms,
        COALESCE(r.bathrooms::INT, 0) AS bathrooms,
        COALESCE(r.sqft_living::INT, s.mode_sqft_living) AS sqft_living,
        COALESCE(r.sqft_lot::INT, s.mode_sqft_lot) AS sqft_lot,
        COALESCE(ROUND(r.floors, 1)::INT, s.mode_floors) AS floors,
        CASE WHEN r.waterfront > 0 THEN 1 ELSE 0 END AS waterfront,
        COALESCE(r.view::INT, s.med_view)::INT AS view,
        CASE WHEN r.condition::INT NOT BETWEEN 1 AND 5 THEN 3 ELSE r.condition::INT END AS condition,
        CASE WHEN r.grade::INT NOT BETWEEN 1 AND 10 THEN 5 ELSE r.grade::INT END AS grade,
        COALESCE(r.sqft_above::INT, s.med_sqft_above) AS sqft_above,
        COALESCE(r.sqft_basement::INT, s.med_sqft_basement)::INT AS sqft_basement,
        CASE WHEN r.yr_built::INT NOT BETWEEN 1900 AND 2015 THEN s.med_yr_built::INT ELSE r.yr_built::INT END AS year_built,
        COALESCE(r.yr_renovated::INT, 0) AS year_renovated,
        CASE WHEN r.zipcode::INT NOT BETWEEN 98001 AND 98199 THEN s.med_zipcode::INT ELSE r.zipcode::INT END AS zipcode,

        -- Newly Added Columns
        COALESCE(r.price::NUMERIC, s.med_price)::INT AS price,
        COALESCE(r.lat::FLOAT, s.med_lat) AS lat,
        COALESCE(r.long::FLOAT, s.med_long) AS long,
        COALESCE(r.sqft_living15::INT, s.med_sqft_living15)::INT AS sqft_living15,
        COALESCE(r.sqft_lot15::INT, s.med_sqft_lot15)::INT AS sqft_lot15

    FROM raw_data r
    CROSS JOIN global_stats s
    
)

SELECT * FROM cleaned ORDER BY date