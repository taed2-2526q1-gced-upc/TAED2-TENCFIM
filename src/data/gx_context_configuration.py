"""Configure a Great Expectations context and checkpoint for data validation."""

import great_expectations as gx

from src.config import INTERIM_DATA_DIR, ROOT_DIR

DATASOURCE_NAME = "pandas"
RAW_DATA_ASSET = "raw_emotions"
CLEAN_DATA_ASSET = "clean_emotions"
RAW_DATA_VALIDATOR = "raw_data_validator"
CLEAN_DATA_VALIDATOR = "clean_data_validator"
EXPECTATIONS_SUITE = "data_quality_validation"
CHECKPOINT = "validation_checkpoint"

if __name__ == "__main__":
    context = gx.get_context(mode="file", project_root_dir=ROOT_DIR)

    data_docs_config = {
        "class_name": "SiteBuilder",
        "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
        "store_backend": {
            "class_name": "TupleFilesystemStoreBackend",
            # Path is relative to the Data Context root
            "base_directory": "data_docs",
        },
    }
    context.update_data_docs_site("local_site", data_docs_config)

    datasource = context.data_sources.add_or_update_pandas(name=DATASOURCE_NAME)

    # raw_asset = datasource.add_parquet_asset(
    #     name=RAW_DATA_ASSET, path=RAW_DATA_DIR / "raw_emotions.parquet"
    # )
    # raw_data_batch_definition = raw_asset.add_batch_definition(name="emotions_data")

    clean_asset = datasource.add_parquet_asset(
        name=CLEAN_DATA_ASSET,
        path=INTERIM_DATA_DIR / "emotions_cleaned.parquet",
    )
    clean_data_batch_definition = clean_asset.add_batch_definition(
        name="emotions_data"
    )

    expectation_suite = gx.ExpectationSuite(EXPECTATIONS_SUITE)
    context.suites.add_or_update(expectation_suite)

    # Explore available expectations at https://greatexpectations.io/expectations/
    # Validate the "text" column
    expectation_suite.add_expectation(gx.expectations.ExpectColumnToExist(column="text"))
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(column="text", type_="str")
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(column="text")
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="text")
    )

    # Validate the "labels" column
    expectation_suite.add_expectation(gx.expectations.ExpectColumnToExist(column="labels"))
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(column="labels", type_="str")
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="labels",
            value_set=[
                "happiness",
                "neutral",
                "sadness",
                "surprise",
                "love",
                "fear",
                "confusion",
                "disgust",
                "desire",
                "shame",
                "sarcasm",
                "anger",
                "guilt",
            ],
        )
    )
    expectation_suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="labels")
    )

    # Only required if the suite already exists so that changes are saved
    expectation_suite.save()

    # Create a validation definition to run our expectations suite
    clean_data_validation_definition = gx.ValidationDefinition(
        name=CLEAN_DATA_VALIDATOR,
        data=clean_data_batch_definition,
        suite=expectation_suite,
    )
    context.validation_definitions.add_or_update(clean_data_validation_definition)

    # Create a checkpoint to run our expectations and compile the results as Data Docs
    action_list = [gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")]
    validation_definitions = [clean_data_validation_definition]

    checkpoint = gx.Checkpoint(
        name=CHECKPOINT,
        validation_definitions=validation_definitions,
        actions=action_list,
        result_format="SUMMARY",
    )

    context.checkpoints.add_or_update(checkpoint)
