import sys
import math
import random

from gurobipy import *

# tested with Python 2.7.6 & Gurobi 6.5

# Inputs para INSTANCIA de 20 TAREFAS
'''machines = [1,2,3]
tarefas =  [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
p =        [ 8, 5, 4, 6, 7, 5, 7, 4, 8, 5, 6, 4, 3, 6, 4, 9, 4, 7, 5, 6]
d =        [ 17, 4, 13, 14, 20, 26, 9, 8, 10, 19, 25, 21, 5, 7, 27, 36, 14, 19, 12, 30]
w =        [ 10, 15, 6, 17, 4, 6, 3, 12, 4, 7, 5, 5, 1, 9, 14, 8, 3, 6, 9, 7]'''

machines = [1,2,3,4,5]
tarefas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200]
p=[9,10,9,10,11,9,12,11,14,9,9,14,12,6,8,11,6,12,14,15,6,15,10,14,12,12,6,13,15,6,11,10,7,13,8,9,7,6,8,10,8,12,11,6,15,10,13,15,12,9,10,12,12,10,14,6,14,14,15,10,15,6,11,14,6,14,10,11,15,15,13,11,10,12,12,13,11,15,13,8,10,13,10,14,11,6,6,15,9,12,10,14,14,12,12,13,12,9,8,14,6,12,7,12,10,9,10,6,6,9,10,14,13,6,13,14,7,15,8,7,8,12,8,12,11,7,10,8,10,14,8,13,8,13,9,10,14,13,6,6,9,6,14,8,11,7,8,13,13,9,8,7,13,6,6,11,7,12,14,12,8,15,14,6,7,12,13,7,6,13,8,15,15,7,10,6,15,14,14,10,8,7,6,9,11,15,10,7,12,10,6,6,11,9,6,8,12,14,10,14]
d=[325,281,308,292,364,210,212,358,307,279,326,317,231,264,209,362,295,316,354,310,272,376,324,309,265,371,246,344,297,305,289,350,221,279,218,368,225,296,331,309,320,246,235,308,336,351,297,374,359,351,258,261,339,317,293,319,338,309,302,267,356,321,256,252,372,224,343,211,275,320,216,320,342,373,320,347,247,370,247,225,238,330,357,306,216,293,305,235,347,252,215,355,263,255,266,251,294,288,304,227,235,320,249,242,258,342,319,301,251,367,248,229,230,259,265,272,271,372,237,366,216,354,250,354,347,211,323,330,256,327,265,302,234,353,370,361,320,303,369,322,320,344,288,366,257,273,280,372,266,218,215,211,290,341,328,272,315,209,365,255,232,251,257,332,271,330,230,243,320,279,317,258,254,284,357,354,287,249,262,240,347,270,300,314,216,313,231,354,309,301,368,258,246,246,229,329,255,342,278,336]
w=[13,17,12,16,19,15,18,23,11,29,30,9,8,26,19,28,18,8,20,5,9,15,15,28,13,26,6,27,16,28,5,20,10,25,30,13,18,25,29,14,18,25,9,22,9,20,23,12,22,22,13,30,9,9,6,5,6,25,28,8,27,7,23,8,9,26,22,25,16,22,19,29,30,21,11,14,23,26,14,23,24,28,8,15,10,16,7,21,6,25,12,15,9,29,30,12,15,26,16,13,25,19,15,30,11,9,29,30,9,15,29,25,15,27,6,19,7,28,28,27,5,6,29,12,24,18,17,11,20,16,5,24,28,11,5,22,16,7,7,21,9,16,30,19,22,17,7,25,19,11,18,10,13,19,26,11,26,10,7,5,18,25,20,10,26,5,13,7,13,25,19,24,10,11,19,21,9,12,23,26,13,20,27,5,26,9,7,29,17,22,27,7,27,7,17,15,11,9,18,19]


#tarefas =  [ 1, 2, 3, 4, 5, 6 ]
#p =        [ 1, 1, 1, 2, 2, 2 ]
#d =        [ 2, 2, 2, 2, 2, 2]
#w =        [ 10, 15, 6, 17, 4, 6]


T=450
tempos =   [t for t in range(T)]

# cria matriz de custos
custos=[]
for j in range(len(tarefas)):
    cb=[]
    for t in range(len(tempos)):
        if (tempos[t]+p[j])<=d[j]:
            cb.append(0)
        else:
            cb.append((tempos[t]+p[j]-d[j])*w[j])
    custos.append(cb)

#print(custos[1])

m = Model('SMP')
#model.Params.UpdateMode = 1

X = {} 

for index1, i in enumerate(machines):
    for index2, j in enumerate(tarefas):
        for index3, t in enumerate(tempos):
            X[i,j,t]=m.addVar(vtype=GRB.BINARY,name="X_{}_{}_{}".format(i,j,t))

m.update()
n_vars = len(m.getVars())


# Uma tarefa e feita por uma maquina
for index2, j in enumerate(tarefas):
    m.addConstr(quicksum(X[i,j,t] for index1, i in enumerate(machines) for index3,t in enumerate(tempos[:(T-p[j-1])]))==1,name="MQ_{}".format(j))


# Um tempo T so pode fazer uma tarefa para cada maquina
expr = LinExpr(0.0)
for index1, i in enumerate(machines):
	for index3, t in enumerate(tempos):
		for index2,j in enumerate(tarefas):
			l1=max(0,t-1-p[j-1]+1)
			for index4,t2 in enumerate(tempos[l1:t]):
				expr.addTerms(1.0, X[i,j,t2])
		m.addConstr(expr<=1,"IT_{}_{}".format(i,t))
		expr.clear()
            
   
# Objective
obj = LinExpr(0.0)
for index1, i in enumerate(machines):
    for index2, j in enumerate(tarefas):
        for index3, t in enumerate(tempos[:T-p[j-1]+1-1]):
            obj.addTerms(custos[j-1][t-1], X[i,j,t])

m.setObjective(obj, GRB.MINIMIZE)
m.update()

m.write("smp.lp")


m.optimize()

for v in m.getVars():
    if abs(v.x)>0.000001:
        print('%s %g' % (v.varName, v.x))
print("j t d w p custo")
for index2,j in enumerate(tarefas):
    for index3, t in enumerate(tempos):
        print(j,t,d[j-1],w[j-1],p[j-1],custos[j-1][t-1])