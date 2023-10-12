from .visitor import Visitor
from typing import List


class ObjectCreationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.object_creation_list = []

    def get_object_creations(self, code: str) -> List[str]:
        tree = self.parser.parse(code.encode())
        root = tree.root_node
        class_body = root.children[0].children[3]
        for child in class_body.children:
            self._get_object_creation(child)
        
        return self.object_creation_list
    
    def _get_object_creation(self, node):
        ## recursion
        
        for child in node.children:
            
            if child.type == 'object_creation_expression':
                self.object_creation_list.append(child.children[1].text.decode())
                
            self._get_object_creation(child)
        
