import pandas as pd

df = pd.read_csv("enhanced_zomato_dataset_clean.csv")

print("Original:", df.shape)

df = df.drop_duplicates(
    subset=["Restaurant_Name","Item_Name","Prices"]
).reset_index(drop=True)


def detect_dietary(item):
    item = str(item).lower()
    nonveg = ["chicken","mutton","fish","prawn","egg","kebab","shawarma","tandoori","bbq"]
    if any(x in item for x in nonveg):
        return "NonVeg"
    return "Veg"

df["dietary_type"] = df["Item_Name"].apply(detect_dietary)

def detect_category(item):
    item = str(item).lower()

    if any(x in item for x in ["pizza","burger","shawarma","wrap","combo","biryani","thali"]):
        return "Main"
    if any(x in item for x in ["fries","nuggets","salad","bread","starter"]):
        return "Side"
    if any(x in item for x in ["shake","lassi","juice","cola","coffee","tea","drink"]):
        return "Beverage"
    if any(x in item for x in ["cake","brownie","dessert","ice cream","sweet"]):
        return "Dessert"

    return "Main"

df["menu_category"] = df["Item_Name"].apply(detect_category)

def price_bucket(p):
    try:
        p = float(p)
    except:
        return "Medium"

    if p < 150: return "Low"
    if p < 300: return "Medium"
    return "High"

df["price_range"] = df["Prices"].apply(price_bucket)

menu_df = df[[
    "Restaurant_Name",
    "City",
    "Place_Name",
    "Cuisine",
    "Item_Name",
    "Prices",
    "Delivery_Rating",
    "Average_Rating",
    "Is_Bestseller",
    "Is_Highly_Rated",
    "Is_Expensive",
    "dietary_type",
    "menu_category",
    "price_range"
]].copy()

menu_df.columns = [
    "restaurant_name",
    "city",
    "place",
    "cuisine_type",
    "item_name",
    "price",
    "delivery_rating",
    "avg_rating",
    "is_bestseller",
    "is_highly_rated",
    "is_expensive",
    "dietary_type",
    "menu_category",
    "price_range"
]

print("Optimized menu:", menu_df.shape)

menu_df.to_csv("csao_menu_full.csv", index=False)

print("Saved: csao_menu_full.csv")