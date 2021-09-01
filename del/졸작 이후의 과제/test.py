import os

print(os.path.abspath(__file__))

print(os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")))
print(os.path.join(os.path.dirname(os.path.abspath(__file__)), "info.csv"))
