"""
Example script demonstrating synthetic data generation with Plexe.

This script creates a synthetic restaurant review dataset that could be used for 
sentiment analysis or restaurant recommendation systems.
"""

from typing import Literal

from pydantic import BaseModel, Field

from plexe import DatasetGenerator


class RestaurantReviewSchema(BaseModel):
    """Schema definition for restaurant reviews dataset."""

    restaurant_name: str = Field(description="Name of the restaurant")
    cuisine_type: str = Field(description="Type of cuisine (Italian, Chinese, Mexican, etc.)")
    price_range: Literal["$", "$$", "$$$", "$$$$"] = Field(
        description="Price category from $ (cheap) to $$$$ (very expensive)"
    )
    location: str = Field(description="City or neighborhood where the restaurant is located")
    rating: float = Field(description="Overall customer rating from 1.0 to 5.0")
    service_rating: int = Field(description="Rating for service quality from 1 to 5")
    food_rating: int = Field(description="Rating for food quality from 1 to 5")


def main():
    # Create dataset generator
    print("Creating synthetic restaurant reviews dataset...")
    dataset = DatasetGenerator(
        description=(
            "Restaurant reviews dataset for sentiment analysis and recommendation systems. "
            "Each record represents a customer review of a restaurant, including a rating."
        ),
        provider="openai/gpt-4o",  # Use your preferred provider
        schema=RestaurantReviewSchema,
    )

    # Generate 20 synthetic records
    print("Generating 20 synthetic reviews...")
    dataset.generate(20)

    # Convert to pandas DataFrame for analysis
    df = dataset.data

    # Display statistics and sample data
    print(f"\nGenerated {len(df)} restaurant reviews")
    print("\nSample reviews:")
    print(df.head(5))

    # Only try to display samples if we have data
    if len(df) > 0:
        for i, row in df.sample(min(5, len(df))).iterrows():
            print(f"\n{'-' * 70}")
            print(f"{row['restaurant_name']} - {row['cuisine_type']} ({row['price_range']}) - {row['location']}")
            print(
                f"Overall: {row['rating']:.1f}/5.0 | Service: {row['service_rating']}/5 | Food: {row['food_rating']}/5"
            )


if __name__ == "__main__":
    main()
