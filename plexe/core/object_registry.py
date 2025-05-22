"""
This module provides a generic Registry pattern implementation for storing and retrieving objects by name or prefix.
"""

import logging
import dataclasses
import copy
from typing import Dict, List, Type, TypeVar, Any


T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Item:
    item: T
    immutable: bool = False


class ObjectRegistry:
    """
    Registry for storing and retrieving objects by name.

    This class implements the Singleton pattern so that registry instances are shared
    across the application. It provides methods for registering, retrieving, and
    managing objects in a type-safe manner.
    """

    _instance = None
    _items: Dict[str, Item] = dict()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectRegistry, cls).__new__(cls)
            cls._items = dict()
        return cls._instance

    @staticmethod
    def _get_uri(t: Type[T], name: str) -> str:
        return f"{str(t)}://{name}"

    def register(self, t: Type[T], name: str, item: T, overwrite: bool = False, immutable: bool = False) -> None:
        """
        Register an item with a given name.

        :param t: type prefix for the item
        :param name: identifier for the item - must be unique within the prefix
        :param item: the item to register
        :param overwrite: whether to overwrite an existing item with the same name
        :param immutable: whether the item should be treated as immutable (not modifiable)
        """
        uri = self._get_uri(t, name)
        if not overwrite and uri in self._items:
            raise ValueError(f"Item '{uri}' already registered, use a different name")
        self._items[uri] = Item(item, immutable=immutable)
        logger.info(f"Registered item '{uri}'")

    def register_multiple(
        self, t: Type[T], items: Dict[str, T], overwrite: bool = False, immutable: bool = False
    ) -> None:
        """
        Register multiple items with a given prefix.

        :param t: type prefix for the items
        :param overwrite: whether to overwrite existing items with the same names
        :param immutable: whether the items should be treated as immutable (not modifiable)
        :param items: dictionary of item names and their corresponding objects
        """
        for name, item in items.items():
            self.register(t, name, item, overwrite=overwrite, immutable=immutable)

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
        return self._items[uri].item if not self._items[uri].immutable else copy.deepcopy(self._items[uri].item)

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
        return {name: item.item for name, item in self._items.items() if name.startswith(str(t))}

    def delete(self, t: Type[T], name: str) -> None:
        """
        Delete an item by name.

        :param t: type prefix for the item
        :param name: the name of the item to delete
        """
        uri = self._get_uri(t, name)
        if uri in self._items:
            del self._items[uri]
        else:
            raise KeyError(f"Item '{uri}' not found in registry")

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

    def get_all_solutions(self) -> List[Dict[str, Any]]:
        """
        Get all solutions tracked during model building.

        This method extracts solution information from the registry, focusing on
        code, performance metrics, and other solution-specific data for checkpointing.

        :return: List of solution data dictionaries
        """
        solutions = []

        # Extract training code and their results
        from plexe.internal.models.entities.code import Code
        from plexe.internal.models.entities.node import Node

        # Get all code objects
        code_items = self.get_all(Code)
        node_items = self.get_all(Node)

        # Build solution data from code and node objects
        for uri, code_obj in code_items.items():
            if isinstance(code_obj, Code):
                # Extract code ID and try to find associated node
                code_id = uri.split("://")[1]
                solution_data = {
                    "code_id": code_id,
                    "code": code_obj.code,
                    "iteration": getattr(code_obj, "iteration", 0),
                }

                # Look for associated node to get performance metrics
                for node_uri, node in node_items.items():
                    if isinstance(node, Node) and node.training_code == code_obj.code:
                        if node.performance:
                            solution_data["performance"] = {
                                "name": node.performance.name,
                                "value": node.performance.value,
                                "comparison_method": getattr(
                                    node.performance.comparator, "comparison_method", "HIGHER_IS_BETTER"
                                ),
                            }
                        break

                solutions.append(solution_data)

        return solutions
