###############################################################################
# General Information
###############################################################################
# Author: ZhongYi Jiang

# operators.py: Operator class, which is used as the building blocks for
# assembling PyTorch expressions with an RNN.

###############################################################################
# Dependencies
###############################################################################

import torch

###############################################################################
# Operators Class
###############################################################################

class Operators:
    """
    The list of valid nonvariable operators may be found in nonvar_operators.
    All variable operators must have prefix 'var_'. Constant value operators
    are fine too (e.g. 3.14), but they must be passed as floats.
    """
    nonvar_operators = [
        '*', '+', '-', '/', '^',
        'cos', 'sin', 'tan',
        'exp', 'ln',
        'sqrt', 'square',
        'c' # ephemeral constant
    ]
    nonvar_arity = {
        '*': 2,
        '+': 2,
        '-': 2,
        '/': 2,
        '^': 2,
        'cos': 1,
        'sin': 1,
        'tan': 1,
        'exp': 1,
        'ln': 1,
        'sqrt': 1,
        'square': 1,
        'c': 0
    }
    function_mapping = {
        '*': 'torch.mul',
        '+': 'torch.add',
        '-': 'torch.subtract',
        '/': 'torch.divide',
        '^': 'torch.pow',
        'cos': 'torch.cos',
        'sin': 'torch.sin',
        'tan': 'torch.tan',
        'exp': 'torch.exp',
        'ln': 'torch.log',
        'sqrt': 'torch.sqrt',
        'square': 'torch.square'
    }

    def __init__(self, operator_list, device):
        """Description here
        """

        # Construct different types of operator list
        self.operator_list = operator_list # list of operators pool for example:['*', '+', '-', '/', '^', 'cos', 'sin', 'var_x', 'exp', 'ln', 'var_y']
        self.constant_operators = [x for x in operator_list if x.replace('.', '').strip('-').isnumeric()] # list of constant operators for example:['1', '2', '3']
        self.nonvar_operators = [x for x in self.operator_list if "var_" not in x and x not in self.constant_operators] # list of non_var operators for example: ['*', '+', '-']
        self.var_operators = [x for x in operator_list if x not in self.nonvar_operators and x not in self.constant_operators] # list of variable operators for example: ['var_x','var_y']
        self.trig_operators = [x for x in self.operator_list if x in ['sin' , 'cos', 'tan']] # list of trigonometric operators for example: ['sin', 'cos']
        self.unary_operators = [x for x in self.operator_list if x in ['ln', 'exp']] # list of unary operators for example: ['ln', 'exp']
        self.const_operators = [x for x in self.operator_list if x in ['c']] # list of constant operators for example :['c']
        self.one_arity_operators = [x for x in self.operator_list if x in ['cos', 'sin', 'tan', 'exp', 'ln', 'sqrt', 'square']] # list of one arity operators for example:['sin','cos', 'exp', 'ln']
        self.__check_operator_list() # Sanity check


        self.device = device

        # Construct data structures for handling arity
        self.arity_dict = dict(self.nonvar_arity, **{x: 0 for x in self.var_operators}, **{x: 0 for x in self.constant_operators}) # list of all operator with its arity for example: {'*':2, 'cos': 1, 'var_x':0}
        self.zero_arity_mask = torch.tensor([1 if self.arity_dict[x]==0 else 0 for x in self.operator_list]).to(device) # list of boolean value which indicate the zero arity operator with 1
        self.nonzero_arity_mask = torch.tensor([1 if self.arity_dict[x]!=0 else 0 for x in self.operator_list]).to(device) # list of boolean value which indicate the nonzero arity operator with 1
        self.variable_mask = torch.Tensor([1 if x in self.var_operators else 0 for x in self.operator_list]) # list of boolean value which indicate the variable operator with 1
        self.nonvariable_mask = torch.Tensor([0 if x in self.var_operators else 1 for x in self.operator_list]) # list of boolean value which indicate the non-variable operator with 1
        self.trigvaribale_mask = torch.Tensor([1 if x in self.trig_operators else 0 for x in self.operator_list]).to(device) # list of boolean value which indicate the trigonometric operators with 1
        self.const_mask = torch.Tensor([1 if x in self.const_operators else 0 for x in self.operator_list]).to(device) # list of boolean value which indicate the constant operators with 1
        self.one_arity_mask = torch.Tensor([1 if x in self.one_arity_operators else 0 for x in self.operator_list]).to(device) # list of boolean value which indicate the one arity operators with 1


        # Contains indices of all operators with arity 2
        self.arity_two = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==2])
        # Contains indices of all operators with arity 1
        self.arity_one = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==1])
        # Contains indices of all operators with arity 0
        self.arity_zero = torch.Tensor([i for i in range(len(self.operator_list)) if self.arity_dict[self.operator_list[i]]==0])
        # Contains indices of all operators that are variables
        self.variable_tensor = torch.Tensor([i for i in range(len(self.operator_list)) if self.operator_list[i] in self.var_operators])
        # Contains indices of all trignometric operators that are variables
        self.trigvaribale_tensor = torch.Tensor([i for i  in range(len(self.operator_list)) if self.operator_list[i] in self.trig_operators])
        # Contains indices of all unary operators that are variables
        self.unaryvaribale_tensor = torch.Tensor([i for i in range(len(self.operator_list)) if self.operator_list[i] in self.unary_operators])
        # Contains indices of all constant operators that are variables
        self.const_tensor = torch.Tensor([i for i in range(len(self.operator_list)) if self.operator_list[i] in self.const_operators])



        # Construct data structures for handling function and variable mappings
        self.func_dict = dict(self.function_mapping)
        self.var_dict = {var: i for i, var in enumerate(self.var_operators)}

    def __check_operator_list(self):
        """Throws exception if operator list is bad
        """
        invalid = [x for x in self.nonvar_operators if x not in Operators.nonvar_operators]
        if (len(invalid) > 0):
            raise ValueError(f"""Invalid operators: {str(invalid)}""")
        return True

    def __getitem__(self, i):
        try:
            return self.operator_list[i]
        except:
            return self.operator_list.index(i)

    def arity(self, operator):
        try:
            return self.arity_dict[operator]
        except NameError:
            print("Invalid operator")

    def arity_i(self, index):
        try:
            return self.arity_dict[self.operator_list[index]]
        except NameError:
            print("Invalid index")

    def func(self, operator):
        return self.func_dict[operator]

    def func_i(self, index):
        return self.func_dict[self.operator_list[index]]

    def var(self, operator):
        return self.var_dict[operator]

    def var_i(self, index):
        return self.var_dict[self.operator_list[index]]

    def __len__(self):
        return len(self.operator_list)
