"""
This module provides a generic Registry pattern implementation for storing and retrieving objects by name or prefix.
"""

from typing import Dict, List, Type, TypeVar


T = TypeVar("T")


class ObjectRegistry:
    """
    Registry for storing and retrieving objects by name.

    This class implements the Singleton pattern so that registry instances are shared
    across the application. It provides methods for registering, retrieving, and
    managing objects in a type-safe manner.
    """

    _instance = None
    _items = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectRegistry, cls).__new__(cls)
            cls._items = {}
        return cls._instance

    @staticmethod
    def _get_uri(t: Type[T], name: str) -> str:
        return f"{str(t)}://{name}"

    def register(self, t: Type[T], name: str, item: T) -> None:
        """
        Register an item with a given name.

        :param t: type prefix for the item
        :param name: identifier for the item - must be unique within the prefix
        :param item: the item to register
        """
        uri = self._get_uri(t, name)
        if uri in self._items:
            raise ValueError(f"Item '{uri}' already registered, use a different name")
        self._items[uri] = item

    def register_multiple(self, t: Type[T], items: Dict[str, T]) -> None:
        """
        Register multiple items with a given prefix.

        :param t: type prefix for the items
        :param items: dictionary of item names and their corresponding objects
        """
        for name, item in items.items():
            self.register(t, name, item)

    def get(self, t: Type[T], name: str) -> T:
        """
        Retrieve an item by name.

        :param t: type prefix for the item
        :param name: the name of the item to retrieve
        :return: The registered item
        :raises KeyError: If the item is not found in the registry
        """
        uri = self._get_uri(t, name)
        if uri not in self._items:
            raise KeyError(f"Item '{uri}' not found in registry")
        return self._items[uri]

    def get_multiple(self, t: Type[T], names: List[str]) -> Dict[str, T]:
        """
        Retrieve multiple items by name.

        :param t: type prefix for the items
        :param names: List of item names to retrieve
        :return: Dictionary mapping item names to items
        :raises KeyError: If any item is not found in the registry
        """
        return {name: self.get(t, name) for name in names}

    def get_all(self, t: Type[T]) -> Dict[str, T]:
        """
        Retrieve all items for a given prefix.

        :param t: type prefix for the items
        :return: Dictionary mapping item names to items
        """
        return {name: item for name, item in self._items.items() if name.startswith(str(t))}

    def clear(self) -> None:
        """
        Clear all registered items.
        """
        self._items.clear()

    def list(self) -> List[str]:
        """
        List all registered item names.

        :return: List of item names in the registry
        """
        return list(self._items.keys())
