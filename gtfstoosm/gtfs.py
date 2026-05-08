"""
GTFS data handling module.

This module provides functionality for parsing and processing GTFS data.
It handles the reading and validation of GTFS feeds.
"""

import logging
import zipfile
import tempfile
import os
from io import BytesIO
from typing import Optional, Union

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

logger = logging.getLogger(__name__)


class GTFSValidationError(Exception):
    """Exception raised when GTFS feed validation fails."""

    pass


class GTFSFeed(BaseModel):
    """Class for storing and querying a GTFS feed."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow Polars DataFrames/LazyFrames
        validate_assignment=True,
    )

    feed_dir: str
    # Store a mix of eager DataFrames and lazy LazyFrames
    tables: dict[str, Union[pl.DataFrame, pl.LazyFrame]] = Field(default_factory=dict)
    name: str | None = None
    required_files: list[str] = Field(
        default_factory=lambda: [
            "agency.txt",
            "stops.txt",
            "routes.txt",
            "trips.txt",
            "stop_times.txt",
        ]
    )
    optional_files: list[str] = Field(
        default_factory=lambda: [
            "shapes.txt",
            "calendar.txt",
            "calendar_dates.txt",
        ]
    )

    # Define required columns for each file (only for files needed for OSM conversion)
    required_columns: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "agency": ["agency_name", "agency_url", "agency_timezone"],
            "stops": ["stop_id", "stop_name", "stop_lat", "stop_lon"],
            "routes": [
                "route_id",
                "route_type",
            ],  # route_short_name OR route_long_name needed but not both
            "trips": ["route_id", "service_id", "trip_id"],
            "stop_times": [
                "trip_id",
                "arrival_time",
                "departure_time",
                "stop_id",
                "stop_sequence",
            ],
            "shapes": [
                "shape_id",
                "shape_pt_lat",
                "shape_pt_lon",
                "shape_pt_sequence",
            ],
        }
    )

    _temp_dir: Optional[tempfile.TemporaryDirectory] = PrivateAttr(default=None)

    def validate_feed(self, strict: bool = False) -> list[str]:
        """
        Validate the GTFS feed structure and contents.

        Args:
            strict: If True, raises GTFSValidationError on any validation failure.
                   If False, returns a list of validation warnings/errors.

        Returns:
            List of validation messages (warnings and errors)

        Raises:
            GTFSValidationError: If strict=True and validation fails
            FileNotFoundError: If the feed file doesn't exist
            zipfile.BadZipFile: If the feed file is not a valid zip
        """
        issues: list[str] = []

        try:
            # Check if feed file exists and is a valid zip
            with zipfile.ZipFile(self.feed_dir, "r") as zip_ref:
                available_files = set(zip_ref.namelist())

                # Validate required files are present
                missing_files: list[str] = []
                for required_file in self.required_files:
                    if required_file not in available_files:
                        missing_files.append(required_file)

                if missing_files:
                    error_msg = (
                        f"Missing required GTFS files: {', '.join(missing_files)}"
                    )
                    issues.append(f"ERROR: {error_msg}")
                    if strict:
                        raise GTFSValidationError(error_msg)

                # Validate file structure (columns) for available files
                for file in available_files:
                    if not file.endswith(".txt"):
                        continue

                    table_name = file[:-4]  # Remove .txt extension

                    # Skip files we don't have column requirements for
                    if table_name not in self.required_columns:
                        continue

                    try:
                        with zip_ref.open(file) as file_obj:
                            # Read just the header to check columns
                            # We only read the first few bytes to avoid full file read
                            df = pl.read_csv(
                                BytesIO(file_obj.read(1024)),
                                infer_schema_length=0,
                                n_rows=1,
                                truncate_ragged_lines=True
                            )

                            # Check for required columns
                            available_columns = set(df.columns)
                            required_cols = set(self.required_columns[table_name])
                            missing_columns = required_cols - available_columns

                            if missing_columns:
                                error_msg = f"{file}: Missing required columns: {', '.join(sorted(missing_columns))}"
                                issues.append(f"ERROR: {error_msg}")
                                if strict:
                                    raise GTFSValidationError(error_msg)

                            # Special check for routes: need at least one name field
                            if table_name == "routes":
                                if (
                                    "route_short_name" not in available_columns
                                    and "route_long_name" not in available_columns
                                ):
                                    error_msg = f"{file}: Must have either route_short_name or route_long_name"
                                    issues.append(f"ERROR: {error_msg}")
                                    if strict:
                                        raise GTFSValidationError(error_msg)

                    except Exception as e:
                        error_msg = f"{file}: Failed to read file - {str(e)}"
                        issues.append(f"ERROR: {error_msg}")
                        if strict:
                            raise GTFSValidationError(error_msg) from e

                # Add success message if no errors found (warnings are ok)
                has_errors = any(issue.startswith("ERROR:") for issue in issues)
                if not has_errors:
                    issues.append("INFO: Basic GTFS structure validation passed")

        except FileNotFoundError as e:
            error_msg = f"GTFS feed file not found: {self.feed_dir}"
            issues.append(f"ERROR: {error_msg}")
            if strict:
                raise GTFSValidationError(error_msg) from e
            raise

        except zipfile.BadZipFile as e:
            error_msg = f"Invalid zip file: {self.feed_dir}"
            issues.append(f"ERROR: {error_msg}")
            if strict:
                raise GTFSValidationError(error_msg) from e
            raise

        return issues

    def validate_referential_integrity(self) -> list[str]:
        """
        Validate referential integrity between GTFS tables.

        This checks that foreign key relationships are valid.
        Note: This is now more expensive with LazyFrames as it requires collection.
        """
        issues: list[str] = []

        if not self.tables:
            issues.append(
                "WARNING: No tables loaded. Call load() before validating referential integrity."
            )
            return issues

        # Helper to get unique values from a table (handles both DF and LF)
        def get_uniques(table_name, col_name):
            if table_name not in self.tables:
                return set()
            table = self.tables[table_name]
            if isinstance(table, pl.LazyFrame):
                return set(table.select(col_name).unique().collect()[col_name].to_list())
            else:
                return set(table[col_name].unique().to_list())

        # Check trips.route_id references routes.route_id
        if "trips" in self.tables and "routes" in self.tables:
            route_ids = get_uniques("routes", "route_id")
            trip_route_ids = get_uniques("trips", "route_id")
            invalid_routes = trip_route_ids - route_ids

            if invalid_routes:
                issues.append(
                    f"ERROR: trips.route_id contains {len(invalid_routes)} invalid references to routes.route_id"
                )

        # Check stop_times.trip_id references trips.trip_id
        if "stop_times" in self.tables and "trips" in self.tables:
            trip_ids = get_uniques("trips", "trip_id")
            stop_time_trip_ids = get_uniques("stop_times", "trip_id")
            invalid_trips = stop_time_trip_ids - trip_ids

            if invalid_trips:
                issues.append(
                    f"ERROR: stop_times.trip_id contains {len(invalid_trips)} invalid references to trips.trip_id"
                )

        # Check stop_times.stop_id references stops.stop_id
        if "stop_times" in self.tables and "stops" in self.tables:
            stop_ids = get_uniques("stops", "stop_id")
            stop_time_stop_ids = get_uniques("stop_times", "stop_id")
            invalid_stops = stop_time_stop_ids - stop_ids

            if invalid_stops:
                issues.append(
                    f"ERROR: stop_times.stop_id contains {len(invalid_stops)} invalid references to stops.stop_id"
                )

        # Check data completeness
        for table_name in ["stop_times", "stops", "routes"]:
            if table_name in self.tables:
                table = self.tables[table_name]
                height = table.select(pl.len()).collect().item() if isinstance(table, pl.LazyFrame) else table.height
                if height == 0:
                    issues.append(f"ERROR: {table_name}.txt is empty")

        if not issues:
            issues.append("INFO: Referential integrity validation passed")

        return issues

    def load(self, validate_feed: bool = True, strict: bool = False) -> None:
        """
        Load a GTFS feed into Polars DataFrames/LazyFrames by extracting to a temp directory.

        Args:
            validate_feed: If True, validates the feed structure before loading
            strict: If True, raises exception on validation failures
        """
        # Validate feed structure first
        if validate_feed:
            validation_issues = self.validate_feed(strict=strict)

            for issue in validation_issues:
                if issue.startswith("ERROR:"):
                    logger.error(issue)
                elif issue.startswith("WARNING:"):
                    logger.warning(issue)
                else:
                    logger.info(issue)

            has_errors = any(issue.startswith("ERROR:") for issue in validation_issues)
            if has_errors and strict:
                raise GTFSValidationError(
                    "Feed validation failed. See logs for details."
                )

        # Create temporary directory for extraction
        self._temp_dir = tempfile.TemporaryDirectory(prefix="gtfstoosm_")
        temp_path = self._temp_dir.name
        logger.debug(f"Extracting GTFS to {temp_path}")

        try:
            with zipfile.ZipFile(self.feed_dir, "r") as zip_ref:
                all_files = zip_ref.namelist()
                
                # Hybrid Strategy:
                # Eager (Small/Metadata): agency, stops, routes, calendar
                # Lazy (Big/Data): stop_times, trips, shapes
                metadata_files = {"agency.txt", "stops.txt", "routes.txt", "calendar.txt", "calendar_dates.txt"}
                
                needed_files = set(self.required_files) | set(self.optional_files)
                for file in all_files:
                    if file in needed_files:
                        logger.debug(f"Extracting {file}")
                        zip_ref.extract(file, temp_path)
                        
                        table_name = file[:-4]
                        file_path = os.path.join(temp_path, file)
                        
                        if file in metadata_files:
                            # Eagerly read small metadata files (Ensures perfect types for math)
                            df = pl.read_csv(file_path, infer_schema_length=None)
                            logger.info(f"Loaded eager metadata: {file} ({df.height:,} records)")
                            self.tables[table_name] = df
                        else:
                            # Lazily scan large data files (Memory mapped from disk)
                            # Use 10k rows for schema inference to ensure speed + accuracy
                            self.tables[table_name] = pl.scan_csv(
                                file_path, 
                                infer_schema_length=10000,
                                rechunk=False
                            )
                            logger.info(f"Initialized lazy scan: {file} (Disk-backed)")

                # Check if we loaded required files
                missing = [
                    f for f in self.required_files if f[:-4] not in self.tables
                ]
                if missing:
                    error_msg = f"Failed to load required files: {', '.join(missing)}"
                    logger.error(error_msg)
                    if strict:
                        raise GTFSValidationError(error_msg)

        except zipfile.BadZipFile as err:
            raise ValueError(
                f"The file at {self.feed_dir} is not a valid zip file"
            ) from err
        except Exception as e:
            logger.error(f"Error loading GTFS feed: {str(e)}")
            if strict:
                raise
            
    def __del__(self):
        """Cleanup temporary directory when the object is destroyed."""
        if hasattr(self, "_temp_dir") and self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
