import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    load messages and categories datasets and merge them using common id
    """
    messages = pd.read_csv("./" + messages_filepath)
    categories = pd.read_csv("./" + categories_filepath)
    df = messages.merge(categories, on="id")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    split column categories into multiple columns and drop the duplicates
    """
    # split categories to multiple columns
    categories = df["categories"].str.split(";", expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # replace value of 2 with 1
    categories = categories.replace(2, 1)
    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    save the df as SQL db
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("DisasterResponse", engine, if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        print(df.related.unique())

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
