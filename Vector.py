import random
class Vector:
    def __init__(self, x):
        self.x = x
    def __add__(self, other):
        try:
            if(type(other) is Vector):
                if len(self.x)!=len(other.x) or len(self.x[0])!=len(other.x[0]): raise IndexError
                for i in range(len(self.x)):
                    for j in range(len(self.x[0])):
                        self.x[i][j] += other.x[i][j]
            else:
                for i in range(len(self.x)):
                    for j in range(len(self.x[0])):
                        self.x[i][j] += other
        except IndexError:
            print("Vector sizes must be equal")

    def __mul__(self, other):
        if(type(other) is Vector):
            if len(self.x[0]) != len(other.x):
                raise ValueError("The number of columns in the first matrix must be equal to the number of rows in the second matrix.")
            result = [[0 for _ in range(len(other.x[0]))] for _ in range(len(self.x))]
            
            for i in range(len(self.x)):          
                for j in range(len(other.x[0])):   
                    for k in range(len(other.x)):   
                        result[i][j] += self.x[i][k] * other.x[k][j]
            
            return Vector(result)
        else:
            for i in range(len(self.x)):
                for j in range(len(self.x[0])):
                    self.x[i][j] *= other

    def __rand__(self):
        for i in range(len(self.x)):
            for j in range(len(self.x[0])):
                self.x[i][j] = random.randint(1, 100)
    def __div__(self, other):
        try:
            if(type(other) is Vector):
                if len(self.x)!=len(other.x) or len(self.x[0])!=len(other.x[0]): raise IndexError
                for i in range(len(self.x)):
                    for j in range(len(self.x[0])):
                        self.x[i][j] /= other.x[i][j]
            else:
                for i in range(len(self.x)):
                    for j in range(len(self.x[0])):
                        self.x[i][j] /= other
        except IndexError:
            print("Vector sizes must be equal")
    def __sub__(self, other):
        try:
            if(type(other) is Vector):
                if len(self.x)!=len(other.x) or len(self.x[0])!=len(other.x[0]): raise IndexError
                for i in range(len(self.x)):
                    for j in range(len(self.x[0])):
                        self.x[i][j] -= other.x[i][j]
            else:
                for i in range(len(self.x)):
                    for j in range(len(self.x[0])):
                        self.x[i][j] -= other
        except IndexError:
            print("Vector sizes must be equal")
    def __str__(self):
        return str(self.x)


# 1D Matrix
# v1 = Vector([0,0,0,0,0,0,0,0])
# v1.__rand__() 
# print(v1)
# v1+5
# print(v1)
# v1-10
# print(v1)
# v1*10
# print(v1)
# v1/5
# print(v1)
# v1//12
# print(v1)

# 2D Matrix
v1 = Vector([[1,2,3],[4,5,6],[7,8,9]])
v2 = Vector([[1,1,1],[1,1,1],[1,1,1]])
v1+v2
print(v1)
v1+120
print(v1)
v3 = Vector([[1,2],[3,4],[5,6]])
v1-v3
print(v1)
v3*5
print(v3)
v5 = v1*v3
print(v5)