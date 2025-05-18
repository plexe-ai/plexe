"""
Example demonstrating dataset augmentation with Plexe:
1. Adding a new column to an existing dataset
2. Adding more rows to an existing dataset
"""

from pydantic import BaseModel, Field

from plexe import DatasetGenerator


class PurchaseSchema(BaseModel):
    """Base schema for purchase data."""

    product_name: str = Field(description="Name of the purchased product")
    category: str = Field(description="Product category")
    price: float = Field(description="Purchase price in USD")
    customer_id: str = Field(description="Unique customer identifier")


class AugmentedSchema(PurchaseSchema):
    """Augmented schema with product recommendation field."""

    recommendation: str = Field(description="Recommended related product")


def main():
    # Step 1: Create base dataset (10 purchase records)
    base_dataset = DatasetGenerator(
        description="E-commerce purchase data with product and customer information",
        provider="openai/gpt-4o",
        schema=PurchaseSchema,
    )
    base_dataset.generate(10)
    df_base = base_dataset.data

    print("Original dataset (10 records):")
    print(df_base.head(3))
    print(f"Shape: {df_base.shape}")

    # Check if we have data before proceeding
    if len(df_base) == 0:
        print("Failed to generate base dataset. Exiting.")
        return

    # Step 2: Add a new column by extending the schema
    augmented_dataset = DatasetGenerator(
        description="E-commerce purchase data with product recommendations",
        provider="openai/gpt-4o",
        schema=AugmentedSchema,
        data=df_base,
    )
    augmented_dataset.generate(0)  # 0 means just transform existing data
    df_column_added = augmented_dataset.data

    print("\nDataset with new 'recommendation' column:")
    print(df_column_added.head(3))
    print(f"Shape: {df_column_added.shape}")

    # Step 3: Add more rows to the augmented dataset
    augmented_dataset.generate(5)  # Add 5 more records
    df_rows_added = augmented_dataset.data

    print("\nFinal dataset with 5 additional records:")
    print(f"Shape: {df_rows_added.shape}")
    print(df_rows_added.tail(3))


if __name__ == "__main__":
    main()
