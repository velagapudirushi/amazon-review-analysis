"""
Utility functions and sample data loader for Amazon Review Sentiment Analysis.
"""

import os
import pandas as pd
from typing import Optional

SAMPLE_REVIEWS_DATA = [
    {
        "Id": 1,
        "ProductId": "B001E4KFG0",
        "UserId": "A3SGXHILPA5V69",
        "ProfileName": "Monique",
        "HelpfulnessNumerator": 1,
        "HelpfulnessDenominator": 1,
        "Score": 5,
        "Summary": "Good Quality Dog Food",
        "Text": "I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells good. My Labrador is finicky and she appreciates this product better than most."
    },
    {
        "Id": 2,
        "ProductId": "B00813GRG4",
        "UserId": "A1D87F6ZCVE5NK",
        "ProfileName": "dll9",
        "HelpfulnessNumerator": 0,
        "HelpfulnessDenominator": 0,
        "Score": 1,
        "Summary": "Not as Advertised",
        "Text": "Product arrived labeled as Jumbo Salted Peanuts... the peanuts were actually small sized unsalted. Not sure if this was an error or if the vendor is deliberately misrepresenting the product. Will not buy again."
    },
    {
        "Id": 3,
        "ProductId": "B000LQOCH0",
        "UserId": "ABXLMWJIXXAIN",
        "ProfileName": "Natalia Corres",
        "HelpfulnessNumerator": 1,
        "HelpfulnessDenominator": 1,
        "Score": 4,
        "Summary": "\"Delight\" says it all",
        "Text": "This is a confection that has been around a few centuries. It is a light, pillowy citrus gelatin with nuts, in this case Filberts. And it is cut into tiny squares and then liberally coated with powdered sugar. Very good treat."
    },
    {
        "Id": 4,
        "ProductId": "B000UA0QIQ",
        "UserId": "A395BORC6FGVXV",
        "ProfileName": "Karl",
        "HelpfulnessNumerator": 3,
        "HelpfulnessDenominator": 3,
        "Score": 2,
        "Summary": "Cough Medicine",
        "Text": "If you are looking for the secret ingredient in Robitussin, I believe I have found it. I got this in addition to the Root Beer Extract. Neither of them had the flavor I was hoping for. Not good at all."
    },
    {
        "Id": 5,
        "ProductId": "B006K2ZZ7K",
        "UserId": "A1UQRSCLF6GW1T",
        "ProfileName": "Michael D. Bigham",
        "HelpfulnessNumerator": 0,
        "HelpfulnessDenominator": 0,
        "Score": 5,
        "Summary": "Great taffy",
        "Text": "Great taffy at a great price. There was a wide assortment of yummy flavors. Delivery was very quick. If your a lover of licorice, this is a must-have!"
    },
    {
        "Id": 6,
        "ProductId": "B006K2ZZ7K",
        "UserId": "ADT0SRK1MGOEU",
        "ProfileName": "Twoapennything",
        "HelpfulnessNumerator": 0,
        "HelpfulnessDenominator": 0,
        "Score": 4,
        "Summary": "Nice Taffy",
        "Text": "I got a wild hair for taffy and ordered this five pound bag. The taffy was all very enjoyable with many flavors: watermelon, root beer, melon, peppermint, etc. Only complaint is the bit of red licorice."
    },
    {
        "Id": 7,
        "ProductId": "B006K2ZZ7K",
        "UserId": "A1D87F6ZCVE5NK",
        "ProfileName": "dll9",
        "HelpfulnessNumerator": 0,
        "HelpfulnessDenominator": 0,
        "Score": 5,
        "Summary": "Great! Just as good as the expensive brands",
        "Text": "This saltwater taffy had great flavors and was very soft and chewy. Each candy is individually wrapped well. None of the candies were stuck to the wrapper, which is unusual for taffy. Wonderful purchase."
    },
    {
        "Id": 8,
        "ProductId": "B000E7L2R4",
        "UserId": "A1P2Y2Q650N501",
        "ProfileName": "Jimmy",
        "HelpfulnessNumerator": 1,
        "HelpfulnessDenominator": 2,
        "Score": 1,
        "Summary": "Tastes stale and terrible",
        "Text": "The package arrived damaged and the oatmeal tasted completely stale and bitter. I could not finish a single spoonful. Strongly recommend avoiding this seller."
    },
    {
        "Id": 9,
        "ProductId": "B001EO5Z3I",
        "UserId": "A2M97OOGY5N817",
        "ProfileName": "Amanda",
        "HelpfulnessNumerator": 4,
        "HelpfulnessDenominator": 5,
        "Score": 3,
        "Summary": "Average taste, high price",
        "Text": "The tea tastes fine and has a mild aroma, but the packaging was damaged and for the price I expected much richer flavor. It is neither great nor terrible, just okay."
    },
    {
        "Id": 10,
        "ProductId": "B001EO5Z3I",
        "UserId": "A3OXHLG6DIBRW8",
        "ProfileName": "C. Miller",
        "HelpfulnessNumerator": 8,
        "HelpfulnessDenominator": 8,
        "Score": 5,
        "Summary": "Best organic green tea on the market",
        "Text": "Smooth, aromatic, and refreshing. No bitter aftertaste even if steeped for several minutes. I drink three cups a day and highly recommend it to any tea enthusiast."
    },
    {
        "Id": 11,
        "ProductId": "B000HDOPZG",
        "UserId": "A281NPSIMI1C2R",
        "ProfileName": "Rebecca",
        "HelpfulnessNumerator": 2,
        "HelpfulnessDenominator": 2,
        "Score": 5,
        "Summary": "Delicious healthy snack",
        "Text": "Crunchy roasted almonds with just the right amount of sea salt. Perfect portion size for work and completely satisfied my cravings without feeling guilty."
    },
    {
        "Id": 12,
        "ProductId": "B000HDOPZG",
        "UserId": "A1Z54EM24Y40LL",
        "ProfileName": "David",
        "HelpfulnessNumerator": 3,
        "HelpfulnessDenominator": 4,
        "Score": 2,
        "Summary": "Way too salty and over-roasted",
        "Text": "These were burnt and excessively salty. My mouth felt parched after eating just a small handful. Not the quality I expected from this brand."
    },
    {
        "Id": 13,
        "ProductId": "B002QWP89S",
        "UserId": "A3OXHLG6DIBRW8",
        "ProfileName": "C. Miller",
        "HelpfulnessNumerator": 5,
        "HelpfulnessDenominator": 5,
        "Score": 3,
        "Summary": "Decent snack, but inconsistent texture",
        "Text": "Some bags were crispy while others in the same bulk box were chewy and soft. Flavor is decent but quality control needs serious improvement."
    },
    {
        "Id": 14,
        "ProductId": "B002QWP89S",
        "UserId": "A2FRJ2S64230",
        "ProfileName": "Sarah T.",
        "HelpfulnessNumerator": 0,
        "HelpfulnessDenominator": 0,
        "Score": 1,
        "Summary": "Horrible chemical odor",
        "Text": "As soon as I opened the bag a noxious plastic smell hit me. I could not bring myself to eat them. Threw the entire carton straight into the trash. Total waste of money."
    },
    {
        "Id": 15,
        "ProductId": "B007JFMH8M",
        "UserId": "A3OXHLG6DIBRW8",
        "ProfileName": "C. Miller",
        "HelpfulnessNumerator": 12,
        "HelpfulnessDenominator": 12,
        "Score": 5,
        "Summary": "Exceptional dark roast coffee pods",
        "Text": "Rich crema, intense aroma, and zero bitterness. Works flawlessly with my machine. This is hands down the best coffee pod available online."
    },
    {
        "Id": 16,
        "ProductId": "B007JFMH8M",
        "UserId": "A1D87F6ZCVE5NK",
        "ProfileName": "dll9",
        "HelpfulnessNumerator": 6,
        "HelpfulnessDenominator": 6,
        "Score": 4,
        "Summary": "Very solid dark roast",
        "Text": "Robust and strong enough to wake anyone up. I wish it came in larger bulk packs, but the flavor is top-notch."
    },
    {
        "Id": 17,
        "ProductId": "B007JFMH8M",
        "UserId": "A37V5D5Q0E2N1",
        "ProfileName": "Elena",
        "HelpfulnessNumerator": 1,
        "HelpfulnessDenominator": 2,
        "Score": 2,
        "Summary": "Bland and watery",
        "Text": "Even on the smallest cup setting, this coffee came out weak and tasted watery. Not bold as described on the package."
    },
    {
        "Id": 18,
        "ProductId": "B000LQOCH0",
        "UserId": "A3OXHLG6DIBRW8",
        "ProfileName": "C. Miller",
        "HelpfulnessNumerator": 4,
        "HelpfulnessDenominator": 4,
        "Score": 5,
        "Summary": "Authentic and fresh",
        "Text": "Arrived sealed fresh and tasted authentic. Delightful texture and delicate rosewater flavor. Absolutely five stars!"
    },
    {
        "Id": 19,
        "ProductId": "B001E4KFG0",
        "UserId": "A105S56ODHGJEK",
        "ProfileName": "Gina",
        "HelpfulnessNumerator": 2,
        "HelpfulnessDenominator": 3,
        "Score": 3,
        "Summary": "Dog likes it, but cans were dented",
        "Text": "The food itself seems healthy and my dog eats it happily, but three of the twelve cans arrived heavily dented. Amazon replaced them quickly though."
    },
    {
        "Id": 20,
        "ProductId": "B000UA0QIQ",
        "UserId": "A3V514T5SVL90U",
        "ProfileName": "Jason",
        "HelpfulnessNumerator": 1,
        "HelpfulnessDenominator": 1,
        "Score": 1,
        "Summary": "Disgusting taste, avoid",
        "Text": "Artificial aftertaste and gave me an upset stomach. Do not buy under any circumstances. Extremely dissatisfied."
    }
]


def load_or_create_sample_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads existing sample data or saves default sample dataset to disk and returns it."""
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = pd.DataFrame(SAMPLE_REVIEWS_DATA)
        if filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)

    # Calculate review length and helpfulness ratio if columns exist
    if "Text" in df.columns and "ReviewLength" not in df.columns:
        df["ReviewLength"] = df["Text"].astype(str).str.split().str.len()

    if "HelpfulnessNumerator" in df.columns and "HelpfulnessDenominator" in df.columns:
        if "HelpfulnessRatio" not in df.columns:
            denom = df["HelpfulnessDenominator"].replace(0, float("nan"))
            df["HelpfulnessRatio"] = df["HelpfulnessNumerator"] / denom

    return df
