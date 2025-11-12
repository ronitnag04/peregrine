"""
Entry point
"""
import sys
from evantrace.parser import Parser

def main():
    print("Hello from evantrace! Beginning Parser test...")
    if len(sys.argv) < 1:
        print("No file to be parsed was provided.")
        
    parser = Parser(sys.argv[1])
    parser.parse()
    
    


if __name__ == "__main__":
    main()
