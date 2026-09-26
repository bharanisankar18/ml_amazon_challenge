import re
import unicodedata


# -----------------------------
# Normalize business name
# -----------------------------
def normalize_name(name):
    if not isinstance(name, str):
        return ""

    # Convert to lowercase
    name = name.lower()

    # Unicode normalization
    name = unicodedata.normalize("NFKC", name)

    # Replace & with and
    name = name.replace("&", " and ")

    # Remove punctuation but keep Unicode letters/numbers
    name = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)

    # Remove extra spaces
    name = re.sub(r"\s+", " ", name).strip()

    return name


# -----------------------------
# Normalize address
# -----------------------------
def normalize_address(address):
    if not isinstance(address, str):
        return ""

    # Lowercase
    address = address.lower()

    # Unicode normalization
    address = unicodedata.normalize("NFKC", address)

    # Replace common separators
    address = address.replace("&", " and ")

    # Remove punctuation
    address = re.sub(r"[^\w\s]", " ", address, flags=re.UNICODE)

    # Remove extra spaces
    address = re.sub(r"\s+", " ", address).strip()

    return address


# -----------------------------
# Add normalized columns
# -----------------------------
def normalize_dataframe(df):

    df = df.copy()

    df["name_normalized"] = df["business_name"].apply(normalize_name)

    df["address_normalized"] = df["business_address"].apply(
        normalize_address
    )

    df["country_normalized"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":

    import pandas as pd

    s1 = pd.read_csv(
        "dataset/train/train_source1.tsv",
        sep="\t"
    )

    s1 = normalize_dataframe(s1)

    print(s1[
        [
            "business_name",
            "name_normalized",
            "business_address",
            "address_normalized"
        ]
    ].head(10))
