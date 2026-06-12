import re

class ReversibleCompiler:
    """
    A compiler that parses standard multi-bit Boolean and arithmetic expressions
    (e.g., '(X + Y) & ~Z') and compiles them into a sequence of bit-level
    reversible gates (XOR, NOT, Toffoli).
    
    Each variable is represented as an 8-bit array of registers (Var_0 to Var_7).
    All carries and intermediate registers are cleaned dynamically during run.
    """
    def __init__(self):
        self.temp_counter = 0

    def get_new_temp_var(self) -> list[str]:
        # Allocates an 8-bit temporary variable
        var_name = f"t{self.temp_counter}"
        self.temp_counter += 1
        return [f"{var_name}_{i}" for i in range(8)]

    def get_variable_regs(self, name: str) -> list[str]:
        # Multi-bit variable maps to 8 registers
        return [f"{name}_{i}" for i in range(8)]

    def tokenize(self, expression: str) -> list[str]:
        # Tokenize variables, operators (+, &, |, ^, ~), and parentheses
        return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[\+\&\|\^\~\(\)]', expression)

    def to_postfix(self, tokens: list[str]) -> list[str]:
        # Shunting-yard algorithm to convert infix to postfix
        # Precedence: ~ (not) > + (add) > & (and) > ^, | (xor, or)
        precedence = {'~': 4, '+': 3, '&': 2, '^': 1, '|': 1}
        output = []
        stack = []
        
        for token in tokens:
            if re.match(r'[A-Za-z_][A-Za-z0-9_]*', token):
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop() # Remove '('
            elif token in precedence:
                while stack and stack[-1] in precedence and precedence[stack[-1]] >= precedence[token]:
                    output.append(stack.pop())
                stack.append(token)
                
        while stack:
            output.append(stack.pop())
            
        return output

    def compile(self, expression: str) -> tuple[list[tuple], list[str]]:
        """
        Compiles the multi-bit expression into a list of reversible CPU instructions.
        Returns (instructions, final_variable_regs).
        """
        self.temp_counter = 0
        tokens = self.tokenize(expression)
        postfix = self.to_postfix(tokens)
        
        stack = [] # Holds lists of 8 register names
        instructions = []
        
        for token in postfix:
            if re.match(r'[A-Za-z_][A-Za-z0-9_]*', token):
                # Variable reference
                stack.append(self.get_variable_regs(token))
            elif token == '~':
                operand = stack.pop()
                temp = self.get_new_temp_var()
                # Compile bitwise NOT
                for i in range(8):
                    instructions.append(('XOR', temp[i], operand[i]))
                    instructions.append(('NOT', temp[i]))
                stack.append(temp)
            elif token == '^':
                op2 = stack.pop()
                op1 = stack.pop()
                temp = self.get_new_temp_var()
                # Compile bitwise XOR
                for i in range(8):
                    instructions.append(('XOR', temp[i], op1[i]))
                    instructions.append(('XOR', temp[i], op2[i]))
                stack.append(temp)
            elif token == '&':
                op2 = stack.pop()
                op1 = stack.pop()
                temp = self.get_new_temp_var()
                # Compile bitwise AND
                for i in range(8):
                    instructions.append(('AND_XOR', temp[i], op1[i], op2[i]))
                stack.append(temp)
            elif token == '|':
                op2 = stack.pop()
                op1 = stack.pop()
                temp = self.get_new_temp_var()
                # Compile bitwise OR: A | B = (A & B) ^ A ^ B
                for i in range(8):
                    instructions.append(('AND_XOR', temp[i], op1[i], op2[i]))
                    instructions.append(('XOR', temp[i], op1[i]))
                    instructions.append(('XOR', temp[i], op2[i]))
                stack.append(temp)
            elif token == '+':
                op2 = stack.pop() # V
                op1 = stack.pop() # U
                temp = self.get_new_temp_var() # T (sum)
                
                # We need carries C_0 to C_8
                # We prefix carry registers unique to this addition operation
                carry_prefix = f"c{self.temp_counter}_add"
                carries = [f"{carry_prefix}_{i}" for i in range(9)]
                
                # 1. Compute addition and carries forward
                for i in range(8):
                    # Sum: T_i = U_i ^ V_i ^ C_i
                    instructions.append(('XOR', temp[i], op1[i]))
                    instructions.append(('XOR', temp[i], op2[i]))
                    instructions.append(('XOR', temp[i], carries[i]))
                    
                    # Carry: C_{i+1} = (U_i & V_i) ^ (C_i & (U_i ^ V_i))
                    instructions.append(('AND_XOR', carries[i+1], op1[i], op2[i]))
                    instructions.append(('XOR', op1[i], op2[i]))
                    instructions.append(('AND_XOR', carries[i+1], carries[i], op1[i]))
                    instructions.append(('XOR', op1[i], op2[i])) # Restore U_i
                    
                # 2. Clean carry registers dynamically (in reverse order from 7 down to 0)
                # This restores all carries C_1..C_8 back to 0!
                for i in range(7, -1, -1):
                    # Uncompute Carry_{i+1}
                    instructions.append(('XOR', op1[i], op2[i]))
                    instructions.append(('AND_XOR', carries[i+1], carries[i], op1[i]))
                    instructions.append(('XOR', op1[i], op2[i]))
                    instructions.append(('AND_XOR', carries[i+1], op1[i], op2[i]))
                
                stack.append(temp)
                
        if len(stack) != 1:
            raise ValueError("Compilation failed: invalid expression tree.")
            
        final_res = stack.pop()
        return instructions, final_res
