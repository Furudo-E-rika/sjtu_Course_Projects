from .visitor import Visitor
from typing import List


class MethodDeclarationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.method_names = []

    def get_method_names(self, code: str) -> List[str]:
        tree = self.parser.parse(code.encode())
        root = tree.root_node
      
        class_body = root.children[0].children[3]

        for child in class_body.children:
            if child.type == 'method_declaration':
                indentifier = child.children[2]
                self.method_names.append(indentifier.text.decode())

        return self.method_names
                        