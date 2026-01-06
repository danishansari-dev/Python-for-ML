def add(a,b):
    """this will add two numbers"""
    return a+b

def subtract(a,b):
    """this will give you a subtraction of two numbers"""
    return a-b

def multiply(a,b):
    """this will multiply two numbers"""
    return a*b

def divide(a,b):
    """this will divide two numbers"""
    if b == 0 :
        return "can't add this ops"
    return a/b

if __name__ == "__main__":
    print("testing.... ")
    print(add(3,4))
    print(subtract(34,55))
