import math
import time
import matplotlib.pyplot as plt

import random
def generate_random_matrix(size):
    return [[random.randint(0, 100) for _ in range(size)] for _ in range(size)]


def matrixgenerator(N, a1, a2, a3):
    return [[a1 if i == j 
            else a2 if abs(i - j) == 1 
            else a3 if abs(i - j) == 2 
            else 0 for j in range(N)] 
            for i in range(N)]

def mtxtimesmtx(matrix1,matrix2):
    resultmtx = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
    if len(matrix1[0])==len(matrix2):
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)): 
                    resultmtx[i][j] += matrix1[i][k] * matrix2[k][j]

    return resultmtx

def mtxplusmtx(matrix1,matrix2):
    return [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]

def mtxminusmtx(matrix1,matrix2):
    return [[matrix1[i][j] - matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]

def matrixDefragmentation(matrix):
    N=len(matrix)
    L=[[matrix[i][j] if i > j else 0 for j in range(N)]for i in range(N)]
    U=[[matrix[i][j] if i < j else 0 for j in range(N)]for i in range(N)]
    D=[[matrix[i][j] if i == j else 0 for j in range(N)]for i in range(N)]
    
    return L,D,U

def Diagonalone(N):
    return [[1 if i==j else 0 for i in range(N)]for j in range(N)]

def matrixinversonlyD(matrix):
    return [[1/matrix[i][j] if i==j else 0 for j in range(len(matrix))] for i in range(len(matrix))]

def mtxminusone(matrix):
    return [[-1*matrix[j][i] for i in range(len(matrix[0]))]for j in range(len(matrix))]

def forward_substitution(A, b):
    n = len(A)
    y = [[0] for _ in range(n)]
    for i in range(n):
        sum_of_products = sum(A[i][j] * y[j][0] for j in range(i))
        y[i][0] = (b[i][0] - sum_of_products) / A[i][i]
    return y

def backward_substitution(A, b):
    n = len(A)

    solution = [[0] for _ in range(n)]

    for i in range(n - 1, -1, -1):
        solution[i][0] = b[i][0]
        for j in range(i + 1, n):
            solution[i][0] -= A[i][j] * solution[j][0]
        solution[i][0] /= A[i][i]

    return solution

def factorizationLU(A,b,x):
    n=len(A)
    U=[[A[j][i] for j in range(n)]for i in range(n)]
    L=Diagonalone(n)
    for i in range(0,n):
        for j in range(0,i):
            L[i][j]=U[i][j]/U[j][j]
            for k in range(j, n):
                U[i][k] -= L[i][j] * U[j][k]

    y=forward_substitution(L,b)
    x=backward_substitution(U,y)
    err_vect=mtxminusmtx(mtxtimesmtx(A,x),b)
    err_norm = math.sqrt(sum(val[0]**2 for val in err_vect))

    return err_norm

def Gauss_Seidl(A, b, x, L,D,U, tolerance=1e-9, max_iterations=100):
    iteration=0
    err_norm=math.inf
    residuum_norms = []
    DplusL=mtxplusmtx(D,L)
    bmg=forward_substitution(DplusL,b)

    while iteration < max_iterations and err_norm>tolerance:
        m=mtxtimesmtx(U,x)
        Mgs=forward_substitution(DplusL,m)
        Mgs1=mtxminusone(Mgs)
        new_x=mtxplusmtx(Mgs1,bmg)
        err_vect=mtxminusmtx(mtxtimesmtx(A,new_x),b)
        err_norm = math.sqrt(sum(val[0]**2 for val in err_vect))
        residuum_norms.append(err_norm)
        x=new_x
        iteration+=1

    return x, residuum_norms, iteration

def Jacobi(A,b,x,L,D,U,tolerance=1e-9,max_iterations=100):
    iteration=0
    err_norm=math.inf
    residuum_norms = []
    D_inv=matrixinversonlyD(D)
    D_inv_minus=mtxminusone(D_inv)
    LplusU=mtxplusmtx(L,U)
    bmj=mtxtimesmtx(D_inv,b)   

    while iteration < max_iterations and err_norm>tolerance:
        m=mtxtimesmtx(LplusU,x)
        Mj=mtxtimesmtx(D_inv_minus,m)
        new_x=mtxplusmtx(Mj,bmj)
        err_vect=mtxminusmtx(mtxtimesmtx(A,new_x),b)
        err_norm = math.sqrt(sum(val[0]**2 for val in err_vect))
        residuum_norms.append(err_norm)
        x=new_x
        iteration+=1

    return x, residuum_norms, iteration

def taskA(e,c,d,f,N=946):
    a1= 5+e
    a2=a3=-1
    A=matrixgenerator(N,a1,a2,a3)
    b = [[math.sin(i * (f+ 1))] for i in range(len(A))]
    x = [[1] for _ in range(len(A))]
    return A,b,x
    
def taskB():
    A,b,x=taskA(3,4,6,3)
    L,D,U=matrixDefragmentation(A)
    start_time = time.time()
    resultg,tabg,numberg=Gauss_Seidl(A,b,x,L,D,U)
    elapsed_time = time.time() - start_time
    print("Metoda Gaussa Seidla")
    print("Czas: "+str(elapsed_time))
    print("Liczba iteraji: "+str(numberg))
    start_time = time.time()
    resultj,tabj,numberj=Jacobi(A,b,x,L,D,U)
    elapsed_time = time.time() - start_time
    print("Metoda Jacobiego")
    print("Czas: "+str(elapsed_time))
    print("Liczba iteraji: "+str(numberj))
    plt.semilogy(range(numberj), tabj)
    plt.semilogy(range(numberg), tabg) 
    plt.legend(["Gauss","Jacobi"])
    plt.title('Zmiana normy residuum w iteracjach metody Jacobiego i Gaussa')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Norma residuum (log)')
    plt.legend()
    plt.show()

def taskC():
    A,b,x=taskA(-2,4,6,3)
    L,D,U=matrixDefragmentation(A)
    start_time = time.time()
    resultg,tabg,numberg=Gauss_Seidl(A,b,x,L,D,U)
    elapsed_time = time.time() - start_time
    print("Metoda Gaussa Seidla")
    print("Czas: "+str(elapsed_time))
    print("Liczba iteraji: "+str(numberg))
    start_time = time.time()
    resultj,tabj,numberj=Jacobi(A,b,x,L,D,U)
    elapsed_time = time.time() - start_time
    print("Metoda Jacobiego")
    print("Czas: "+str(elapsed_time))
    print("Liczba iteraji: "+str(numberj))
    plt.semilogy(range(numberj), tabj) 
    plt.semilogy(range(numberg), tabg) 
    plt.title('Zmiana normy residuum w iteracjach metody Jacobiego i Gaussa')
    plt.xlabel('Numer iteracji')
    plt.ylabel('Norma residuum (log)')
    plt.show()

def taskD():
    A,b,x=taskA(-2,4,6,3)
    start_time = time.time()
    err=factorizationLU(A,b,x)
    elapsed_time = time.time() - start_time
    print("Metoda Faktoryzacji LU")
    print("Norma residuum "+str(err))
    print("Czas: "+str(elapsed_time))

def taskE():
    TimeG=[]
    TimeJ=[]
    TimeLU=[]
    size=[100,500,1000,2000,3000]
    size2=[100,200,300,400,500]
    for i in size:
        A,b,x=taskA(3,4,6,3,i)
        L,D,U=matrixDefragmentation(A)
        start_time = time.time()
        resultg,tabg,numberg=Gauss_Seidl(A,b,x,L,D,U)
        elapsed_time = time.time() - start_time
        TimeG.append(elapsed_time)
        start_time = time.time()
        resultj,tabj,numberj=Jacobi(A,b,x,L,D,U)
        elapsed_time = time.time() - start_time
        TimeJ.append(elapsed_time)
        start_time = time.time()
        err=factorizationLU(A,b,x)
        elapsed_time = time.time() - start_time
        TimeLU.append(elapsed_time)
    plt.plot(size2, TimeG) 
    plt.plot(size2, TimeJ) 
    plt.plot(size2, TimeLU) 
    plt.title('Zaleznosc czasu od wielkosci N')
    plt.xlabel('Rozmiar (N)')
    plt.ylabel('Czas (s)')
    plt.legend(["Gauss","Jacobi","Faktoryzacja LU"])
    plt.show()
    

taskE()    
