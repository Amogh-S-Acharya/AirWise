import pandas as pd

DATA_PATH = "data/raw/city_day.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    return df


def prepare_city_data(df, city="Delhi"):

    # Select one city
    city_df = df[df["City"] == city].copy()

    # Sort chronologically
    city_df = city_df.sort_values("Date")

    # Remove rows where AQI is unavailable
    city_df = city_df.dropna(subset=["AQI"])

    return city_df


if __name__ == "__main__":

    df = load_data()

    print("\n========== ORIGINAL DATASET ==========")
    print("Shape:", df.shape)

    print("\n========== PREPARING DELHI DATA ==========")

    delhi_df = prepare_city_data(df, "Delhi")

    print("Delhi rows with AQI:", len(delhi_df))

    print("\n========== DATE RANGE ==========")
    print("Start:", delhi_df["Date"].min())
    print("End:", delhi_df["Date"].max())

    print("\n========== MISSING VALUES ==========")
    print(delhi_df.isnull().sum())

    print("\n========== FIRST 5 ROWS ==========")
    print(delhi_df.head())

    print("\n========== LAST 5 ROWS ==========")
    print(delhi_df.tail())