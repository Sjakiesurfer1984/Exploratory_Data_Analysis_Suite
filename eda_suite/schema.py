from typing import List, Dict

class SchemaManager:
    """Manages a mapping between original and display column names."""

    def __init__(self, initial_columns: List[str]):
        self._original_columns = initial_columns
        self._mapping = {col: col for col in initial_columns}
        self._reverse_mapping = {col: col for col in initial_columns}

    def set_mapping(self, mapping_dict: Dict[str, str]):
        """Sets or updates the column name mapping."""
        self._mapping.update(mapping_dict)
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}

    def get_display_name(self, original_name: str) -> str:
        """Gets the display name for an original column name."""
        return self._mapping.get(original_name, original_name)

    def get_original_name(self, display_name: str) -> str:
        """Gets the original column name for a display name."""
        return self._reverse_mapping.get(display_name, display_name)