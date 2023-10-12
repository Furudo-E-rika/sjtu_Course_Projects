from .visitor import Visitor
from typing import List, DefaultDict


class MethodInvocationVisitor(Visitor):
    def __init__(self):
        super().__init__()
        self.method_invocation_list = []
        self.local_variable_declaration_list = []

    def get_method_invocations(self, code: str) -> List[str]:
        # first get all method invocations
        # second if the object of a method invocation is Identifier, replace it by Type
        tree = self.parser.parse(code.encode())
        root = tree.root_node
        class_body = root.children[0].children[3]
        for child in class_body.children:
            self._get_method_invocation(child)
            self._get_local_variable_declaration(child)

        return self.replace_identifier_by_type()
        
    
    def _get_method_invocation(self, node):
        ## get all method invocations recursively
        for child in node.children:
            
            if child.type == 'method_invocation':
                self.method_invocation_list.append(child)
            
            self._get_method_invocation(child)

    def _get_local_variable_declaration(self, node):
        ## get all local variable declarations recursively
        for child in node.children:
            if child.type == 'local_variable_declaration':
                self.local_variable_declaration_list.append(child)
            
            self._get_local_variable_declaration(child)

    def replace_identifier_by_type(self):
        #  if the object of a method invocation is Identifier, replace it by Type

        #  construct a dictionary to store the type of each identifier
        type_dict = {}
        API_sequence_list = []
        for node in self.local_variable_declaration_list:
            type = node.children[0]
            identifier = node.children[1].children[0]
            type_dict[identifier.text.decode()] = type.text.decode()
        
        for node in self.method_invocation_list:
            if node.children[0].type == 'identifier':
                identifier_text = node.children[0].text.decode()
                total_text = node.text.decode()
                total_text = total_text.replace(identifier_text, type_dict[identifier_text])
                total_text = total_text.split('(')[0] 
                
            else:
                total_text = node.text.decode()
                total_text = total_text.split('(')[0]
            
            API_sequence_list.append(total_text)

        return API_sequence_list
        
