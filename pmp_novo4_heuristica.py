from gurobipy import *
#import numpy as np
#import statistics as stt
import random as rd
import time

class job_class:
    def __init__(self,idi, p1, d1, w1):
        self.idi = idi
        self.p = p1
        self.d = d1
        self.w = w1
        self.dw = d1*w1
        
class Cmaquina:
    id2 = 0
    T1 = 1
    T2 = 0
    AT = 0
    VAT = 0.0


class cArc:
    t0 = 0
    t1 = 0
    j = -1
    c = 0.0
    rc = 0.0

class Cust:
    custo = 99999.0
    pai = None
    job = None
    
def esc(a,b):
	soma = 0
	for i in range(len(a)):
		soma += a[i]*b[i]
	return soma;
    
def prod_cte(a,b):
    soma = []
    for el in b:
        soma.append(a*el)
    return soma;

def sum_lista(a,b):
    soma = []
    for i in range(len(a)):
        soma.append(a[i]+b[i])
    return soma;

def media(l):
    return sum(l)/len(l);

    
def problema_mestre():
    global L,R,m, coef_obj
    
    N_sols=len(sol)
    if N_sols > N_maquinas:
        # Calculo do coeficient de D
        dr1 = []
        s=sol[-1]
        for j in range(len(tarefas)):
            a=s[j]
            b=esc(a,cr)
            dr1.append(b)
        
        # Calculo do coeficiente da funcao objetivo
        vobj=0
        for j in tarefas:
            c = custos[j-1]
            d = s[j-1]
            vobj = vobj + esc(c,d)
            
        coef_obj.append(vobj)
        col = Column()
        coefs=dr1+[1]
        
        for k in range(len(tarefas)):
            col.addTerms(coefs[k],R[k+1])
        k=len(tarefas)
        col.addTerms(coefs[k],R[k+1])
            
        L[N_sols-1]=m.addVar(lb=0.0, ub=1.0, obj=vobj, vtype=GRB.CONTINUOUS, name="L_{}".format(N_sols-1), column=col)
        
        m.update()
        N_vars=len(m.getVars())
        
        m.optimize()
        
        # preco sombra do problema em Lambda
        pis=[]
        pis=m.getAttr("Pi", m.getConstrs())
        
        ls=[]
        for v in m.getVars():
            ls.append(v.x)
            
        func_obj=m.ObjVal
    else:
        func_obj, pis, N_vars, ls = problema_mestre_inicial()
    
    return func_obj, pis, N_vars, ls
        

def problema_mestre_inicial():
    global L,R,m, coef_obj
    
    N_sols=len(sol)
    # Calculo do vetor DR  [N_tarefas X N_Sol]
    temp = []
    dr2 = []
    dr1 = []
    for s in sol:
        for j in tarefas:
            a=s[j-1]
            b=esc(a,cr)
            dr1.append(b)
        dr2.append(dr1)
        dr1=[]
    dr=[]
    a=[]
    for i in range(len(tarefas)):
        for k in range(N_sols):
            a.append(dr2[k][i])
        dr.append(a)
        a=[]
    
    
    # Coeficientes da restricao Soma L = 1
    dr21=[1 for t in range(N_sols)]
    
    # Calculo do vetor P [ 1 X N_Sol ]
    P=[]
    vobj=0
    for s in sol:
        for j in tarefas:
            c = custos[j-1]
            d = s[j-1]
            vobj = vobj + esc(c,d)
        P.append(vobj)
        vobj=0
        
    coef_obj = P[:]
    
    
    # PROBLEMA EM LAMBDA
    m = Model("mip1")

    L = {} 

    # Adiciona variavei e as dispoe em um vetor
    for index1, k in enumerate(range(N_sols)):
        L[k]=m.addVar(vtype=GRB.CONTINUOUS,name="L_{}".format(k))
    m.update()
    N_vars=len(m.getVars())
    vec_cells_d=[]
    for index1, k in enumerate(range(N_sols)):
        vec_cells_d.append(L[k])

    #Define funcao objetivo
    m.setObjective(LinExpr(P,vec_cells_d), GRB.MINIMIZE)
    m.update()

    R = {}
    #Adiciona restricoes [DR x L = 1]
    for index2, j in enumerate(tarefas):
        R[j] = m.addConstr(LinExpr(dr[j-1],vec_cells_d) == 1, "cr_{}".format(j))
    
    #Adiciona restricoes [1 x L = 3]
    R[len(tarefas)+1]=m.addConstr(LinExpr(dr21,vec_cells_d)== N_maquinas, "cr_21")

    
    #otimiza chegando a uma solucao otimizada
    m.update()
    m.optimize()

    # preco sombra do problema em Lambda
    pis=[]
    pis=m.getAttr("Pi", m.getConstrs())
    
    ls=[]
    for v in m.getVars():
        ls.append(v.x)
        
    func_obj=m.ObjVal
    

    return func_obj, pis, N_vars, ls;

def sub_problema(pis):
    cust_n = [Cust() for k in range(T+1)]
    cust_n[0].custo=0
    for t in range(T):
        ab=range(len(nodes[t]))
        for j in ab:
            e=nodes[t][j    ]
            if e.j != -1:
                cr = e.c - pis[e.j-1]
            else:
                cr = 0
                
            if cust_n[e.t1].custo > (cust_n[e.t0].custo + cr):
                cust_n[e.t1].custo = cust_n[e.t0].custo + cr 
                cust_n[e.t1].pai = e.t0
                cust_n[e.t1].job = e.j
      
    func_obj = cust_n[-1].custo - pis[-1]
    
    
    ordem=[]  
    t1=len(cust_n)-1    
    t0=cust_n[t1].pai
    b=[t0,cust_n[t1].job,cust_n[t1].custo]
    ordem.append(b)
    while t0>0:
        t1=t0
        t0=cust_n[t1].pai
        b=[t0,cust_n[t1].job,cust_n[t1].custo]
        ordem.append(b)
   
    sol = [[0 for j in range(T)] for i in range(len(tarefas))]
   
    for k in ordem:
        j=k[1]
        t=k[0]
        if j != -1:
            sol[j-1][t]=1
    return func_obj, sol;


def cria_rede():    
    global nodes
   
    # Creat Network
    nodes = []

    # waiting arcs
    for k in range(T):
        temp = cArc()
        temp.t0 = k
        temp.t1 = k + 1
        temp2 = []
        temp2.append(temp)
        nodes.append(temp2)

    # outros arcos
    for t in range(T):
        for j in tarefas:
            if (t + param[j-1].p) < (T + 1):
                temp = cArc()
                temp.t0 = t
                temp.t1 = t + param[j-1].p
                temp.j = j
                if ( t  +  param[j-1].p ) > param[j-1].d:
                    temp.c = (t + param[j-1].p - param[j-1].d)*param[j-1].w
                nodes[t].append(temp)
                
    '''for a in nodes:
        for b in a:
            print(b.t0,b.t1,b.c,b.j)'''
         
 
def ConstructRandomSolution(n_ele_rangd,nmq):
    SOLUC = [[] for p in range(nmq)]
    tf_faltantes=[p for p in tarefas]
    Result = [ 0. for p in range(nmq)]
    while len(tf_faltantes) > 0:
        TempMaquina=Cmaquina()
        if (len(tarefas) - len(tf_faltantes)) >= nmq:
            Vect_T2 = [SOLUC[p][-1].T2 for p in range(nmq)]
            mq = Vect_T2.index(min(Vect_T2))
            TempMaquina.T1 = SOLUC[mq][-1].T2
        else:
            mq = len(tarefas) - len(tf_faltantes)
        lista_faltante = [ param[k-1] for k in tf_faltantes] 
        candidate_set = sorted(lista_faltante, key = lambda x: x.d/x.w) [:n_ele_rangd]
        n_rand =rd.randint(0,min(n_ele_rangd,len(tf_faltantes))-1)
        s = candidate_set[n_rand].idi
        TempMaquina.T2= TempMaquina.T1+param[s-1].p
        TempMaquina.AT=TempMaquina.T2-param[s-1].d;
        if TempMaquina.AT < 0:
            TempMaquina.AT=0
        TempMaquina.VAT = TempMaquina.AT*param[s-1].w;
        TempMaquina.id2 = s
        SOLUC[mq].append(TempMaquina)
        Result[mq] += TempMaquina.VAT 
        tf_faltantes.pop(tf_faltantes.index(s))
    return Result,SOLUC;

def RandomMultiStart(nmq):
    fobj = 9999999.
    delta = 1.
    k = 1
    sx_f=[]
    while delta > 0.01 and k < 5000:  
        nlista=divmod(k,5)[1]+1
        res,sx = ConstructRandomSolution(nlista,nmq)
        if sum(res) < fobj:
            res_f = res[:]
            sx_f = sx[:]
            fobj = sum(res_f)
            nll = nlista 
            if k==1:
                delta = sum(res_f)
            else:
                delta = abs(delta - sum(res_f))
        k += 1
    return res_f,sx_f
   
    
            
    
########################################################################
###  Problema de Scheduling de Maquinas Paralelas Identicas ############
########################################################################

tempo1 = time.time()
tarefas=[]
p=[]
d=[]
w=[]
f = open('inputs.txt')
a = f.readline()
a = f.readline()
elem = a.split()
T = int(elem[0])
N_maquinas = int(elem[1])
iter_max = int(elem[2])


a = f.readline()
for line in f:
    elem = line.split()
    tarefas.append(int(elem[0]))
    p.append(int(elem[1]))
    d.append(int(elem[2]))
    w.append(int(elem[3]))
    

tempos =   [t for t in range(T)]
cr =   [1 for t in range(T)]

param = [job_class(tarefas[i],p[i],d[i],w[i]) for i in range(len(tarefas))]
valores =[]

# cria solucao inicial Heuristica RandomMultiStart(N_maquinas)
SOL = [[] for p in range(N_maquinas)]
print("Calculo da Primeira Solucao Heuristica")
res,SOL = RandomMultiStart(N_maquinas)
print("Funcao Ojetivo:  {}  ".format(sum(res))) 
tempo3 = time.time()


for mq,k in enumerate(SOL):
    print("Maquina {}".format(mq))
    print("JOB  T1\n")
    for p in k:
        print("{} {}".format(p.id2,p.T1))
     
# cria matriz de custos
custos=[]
for j in range(len(tarefas)):
    cb=[]
    for t in range(len(tempos)):
        if (tempos[t]+param[j].p)<=param[j].d:
            cb.append(0)
        else:
            cb.append((tempos[t]+param[j].p-param[j].d)*param[j].w)
    custos.append(cb)


s=[]
s2=[]
for k in SOL:
    stemp=[ 0 for i in range(len(tarefas))]
    stemp2=[ 0 for i in range(len(tarefas))]
    for p in k:
        stemp[p.id2-1]=p.T1
        stemp2[p.id2-1]=p.T2
    s.append(stemp)
    s2.append(stemp2)

   
sol=[] 
b=[] 
si=[]
for a in s:
    for i in range(len(tarefas)):
        for j in range(T):
            if a[i]==(j+1):
                b.append(1)
            else:
                b.append(0)
        si.append(b)
        b=[]
    sol.append(si)
    si=[]

# cria rede
print("Cria Rede")
cria_rede()



temp=[]
vect=[]
solucao=[]
n_var_vec=[]
LS=[]
k = 1
obj_sub = -1

while ((obj_sub < -0.001)&(k<iter_max)):
    print("\n======================================================================================\n")
    print("Problema Mestre {}".format(k))
    print("Numero de Colunas: {}\n".format(len(sol)))
    obj_master, pis, N_vars, LS = problema_mestre()
    print("\n\nFuncao Objetivo Master: {}".format(obj_master))
    print("\nPrecos Sombra: ",pis)
    print("\n\nSub-Problema")
    obj_sub, sol_sub = sub_problema(pis)
    print("\n\nFuncao Objetivo Sub: {}".format(obj_sub))
    temp.append(obj_master)
    temp.append(obj_sub)
    temp.append(sum(pis[:-2]))
    temp.append(pis[-1])
    temp.append(obj_sub)
    temp.append(time.time()-tempo3)
    
    vect.append(temp)
    temp=[]
    sol.append(sol_sub)
    n_var_vec.append(N_vars)
    k += 1
tempo2 = time.time()              

print("\n\n\n\nIteracao   Upper        Lower         SP   Tempos(s)   Lambda   coef_obj")
for i in range(k-1):
    print("{}     {:.2f}        {:.2f}        {:.2f}      {:.4f}".format(i+1,vect[i][0],vect[i][0]+vect[i][1], vect[i][1],vect[i][-1])) 

    

sol_fin=[]
for j in range(len(tarefas)):
    sumj = [0 for t in range(T)]
    for k in range(len(LS)):
        lin_j = sol[k][j]
        lin_prd = prod_cte(LS[k],lin_j)
        sumj = sum_lista(sumj,lin_prd)
    sol_fin.append(sumj)


    
    
print("\n\n\Solucao em Lambda:")
m.write("mestre.lp")
print("lambdas")
print(LS)
print("coefs")
print(coef_obj) 

print("\n\n\Solucao em tj:")
for p in sol_fin:
    print(p)


for p in sol_fin:
    print(p)

m4 = Model("Separa_Maquinas")
X = {} 

maquinas_set = [p+1 for p in range(N_maquinas)]

    
print("\n\n\Solucao Resumo:")
print("\n Tarefa  Tempo  %Tarefa") 
for j in range(len(tarefas)):
    for t in range(T):
        if sol_fin[j][t] != 0:
            print("{} {} {}".format(j+1,t,sol_fin[j][t]))
            for index2, k in enumerate(maquinas_set):
                X[j,t,k]=m4.addVar(vtype=GRB.CONTINUOUS,name="X_j_{}_t_{}_m_{}".format(j+1,t,k))
m4.update()            

m4.setObjective(1.0, GRB.MINIMIZE)
m4.update()

R = {}
for j in range(len(tarefas)):
    for t in range(T):
        if sol_fin[j][t] != 0:
            m4.addConstr(quicksum(X[j,t,k] for index2, k in enumerate(maquinas_set)) == sol_fin[j][t], "cr_{}_{}".format(j,t))
m4.update()


print("capacidade")
expr = LinExpr(0.0)
for index2, k in enumerate(maquinas_set):
    for t in range(T):
        achou = 0
        for j in range(len(tarefas)):
            l1=int(max(0,t-param[j].p))
            for t2 in tempos[l1:t]:
                if sol_fin[j][t2] != 0:  
                    achou = 1
                    expr.addTerms(1.0, X[j,t2,k])
        if achou==1:
            m4.addConstr(expr<=1,"IT_{}_{}_{}".format(j,t,k))
            expr.clear()


m4.update()
    
    
    
#otimiza chegando a uma solucao otimizada

m4.write("separa_20.lp")
#m4.optimize

'''for v in m4.getVars():
    if abs(v.x)>0.0001:
        print('%s %g' % (v.varName, v.x))'''


system("gurobi_cl ResultFile=separa_20.sol separa_20.lp")
system("type separa.sol > lheu.txt")


print("Tempo de processamento (min)")
print((tempo2-tempo1)/60)