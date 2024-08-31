import numpy as np
from pydantic import BaseModel, Field
from raptor_library.embedding import embed_model


class Node(BaseModel):
    text: str
    layer: int = Field(default=0)
    children: set["Node"] = Field(default_factory=set)

    @property
    def embeddings(self) -> np.ndarray:
        return embed_model.encode([self.text])

    def add_child(self, child: "Node") -> None:
        self.children.add(child)

    def get_children(self) -> set["Node"]:
        return self.children

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class Tree(BaseModel):
    root_nodes: list[Node] = Field(default_factory=list)

    def add_node(self, parent: Node, child: Node) -> None:
        parent.add_child(child)

    def traverse(self, nodes: list[Node] | None = None) -> list[str]:
        if nodes is None:
            nodes = self.root_nodes
        all_nodes = []
        for node in nodes:
            all_nodes.append(node.text)
            all_nodes.extend(self.traverse(list(node.get_children())))
        return all_nodes

    def find_node(self, node_text: str, nodes: list[Node] | None = None) -> Node | None:
        if nodes is None:
            nodes = self.root_nodes
        for node in nodes:
            if node.text == node_text:
                return node
            found_node = self.find_node(node_text, list(node.get_children()))
            if found_node:
                return found_node
        return None

    def get_leaves(self) -> list[Node]: ...
