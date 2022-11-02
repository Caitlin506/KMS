import numpy as np
from numba import njit
from tqdm import tqdm

n=7
a=0.38
f=10000
R=0.38
L=1.3*a*(n-1)
m=40
epsilon=1
k=8.31*10**(-3)
T0=170
tau = 1e-3

N=n**3
b0=np.array([a,0.,0.])
b1=np.array([a/2,a*np.sqrt(3)/2,0.])
b2=np.array([a/2,a*np.sqrt(3)/6,a*np.sqrt(2/3)])

#Wyznaczanie położeń początkowych
#file1=open("data1/starting_position.txt","w",encoding="utf-8")
#file1.write(str(N)+"\n\n")
ri0=[]
for i0 in range(n):
    for i1 in range(n):
        for i2 in range(n):
            ri=(i0-(n-1)/2)*b0+(i1-(n-1)/2)*b1+(i2-(n-1)/2)*b2
            #file1.write("Ar"+"\t"+str(ri[0])+"\t"+str(ri[1])+"\t"+str(ri[2])+"\n")
            ri0.append(np.array([ri[0],ri[1],ri[2]]))
            
#file1.close()

#wyznaczanie pędów początkowych
sumP=0
pi0=[]
for i in range(N):
    Ex=-1/2*k*T0*np.log(np.random.rand())
    Ey=-1/2*k*T0*np.log(np.random.rand())
    Ez=-1/2*k*T0*np.log(np.random.rand())
    px=np.sqrt(2*m*Ex)
    py=np.sqrt(2*m*Ey)
    pz=np.sqrt(2*m*Ez)
    a=np.random.rand()
    if a<0.5:
        px=-px
    b=np.random.rand()
    if b<0.5:
        py=-py
    c=np.random.rand()
    if c<0.5:
        pz=-pz
    pi0.append(np.array([px,py,pz]))
    sumP=sumP+np.array([px,py,pz])
    
pi0p=pi0-1/N*sumP

#file2=open("data1/starting_momentum.txt","w",encoding="utf-8")
#for i in range(N):
    #file2.write(str(pi0p[i][0])+"\t"+str(pi0p[i][1])+"\t"+str(pi0p[i][2])+"\n")
            
#file2.close()

#Potencjały
#def V_p(ri,rj):
#    norm=np.linalg.norm(ri-rj)
#    my6=(R/norm)*(R/norm)*(R/norm)*(R/norm)*(R/norm)*(R/norm)
#    vp = epsilon*(my6*my6-2*my6)
#    return vp
#def V_s(ri):
#    norm=np.linalg.norm(ri)
#    if norm<L:
#        return 0
#    else:
#        return 1/2*f*(norm-L)**2
@njit
def V_system(ri,R,f,L):
    systemV = 0
    for i in range(N):
        for j in range(i):
            norm2=np.linalg.norm(ri[i]-ri[j])
            my6=(R/norm2)*(R/norm2)*(R/norm2)*(R/norm2)*(R/norm2)*(R/norm2)
            vp = epsilon*(my6*my6-2*my6)
            systemV=systemV+vp
        norm=np.linalg.norm(ri[i])
        if norm<L:
            continue
        else:
            vs=1/2*f*(norm-L)**2
            systemV=systemV+vs
    return systemV

systemV = V_system(np.array(ri0),R,f,L)
#print(systemV)

#Siły działające na cząsteczkę
@njit
def F_p(ri,rj):
    norm=np.linalg.norm(ri-rj)
    my6=(R/norm)*(R/norm)*(R/norm)*(R/norm)*(R/norm)*(R/norm)
    fp=12*epsilon*(my6*my6-my6)*(ri-rj)/(norm*norm)
    return fp
@njit
def F_s(ri):
    norm=np.linalg.norm(ri)
    # if norm>=L: dodany przy implementacji      
    ret=f*(L-norm)*ri/norm
    return ret
@njit
def Force_i (ri):
    forces = np.empty((0,3), float)
    for i in range(N):
        suma=np.array([0.,0.,0.])
        for j in range(N):
            if j!=i:
                suma=suma+F_p(ri[i],ri[j])
        if (np.linalg.norm(ri[i]))<L:
            forces= np.append(forces, np.array([[suma[0],suma[1],suma[2]]]), axis=0)
        else:
            fi = suma+F_s(ri[i])
            forces= np.append(forces, np.array([[fi[0],fi[1],fi[2]]]), axis=0)
    return forces

forces = Force_i(np.array(ri0))

#Ciśnienie na ścianki
@njit
def get_funsum(ri):
    funsum=0
    for i in range(N):
        if (np.linalg.norm(ri[i]))<L:
            funsum=funsum
        else:
            funsum=funsum+np.linalg.norm(F_s(ri[i]))
    return funsum
@njit
def getP(funsum):
    P = 1/(4*np.pi*L*L)*funsum
    return P

getP(get_funsum(np.array(ri0)))

#Hamiltonian
#Hamiltonian = 0
#for i in range(N):
#    norm=np.linalg.norm(pi0[i])
#    ham = norm*norm/(2*m)
#    Hamiltonian=Hamiltonian+ham
#Hamiltonian=Hamiltonian+systemV
#print(Hamiltonian)

def move (ri0,Fi0,pi0):
    pihalf = pi0+0.5*Fi0*tau
    newri = ri0+1/m*pihalf*tau
    newFi = np.array(Force_i(newri))
    newpi = pihalf+1/2*newFi*tau
    return newri,newFi,newpi

@njit
def kinetic(pi):
    suma=0
    for p in pi:
        norm=np.linalg.norm(p)
        suma = suma + norm*norm/(2*m)
        temp = 2/(3*N*k)*suma
    return suma,temp

def energies_file(a,b,c,d,e):
    with open("data1/energies170.txt", "a") as f:
        f.write(str(a)+'\t')
        f.write(str(b)+'\t')
        f.write(str(c)+'\t')
        f.write(str(d)+'\t')       
        f.write(str(e)+'\n') 
        
def movement(ri,Fi,pi):
    file2=open("data1/movement170.txt","w")
    for i in tqdm(range(10000)):
        ri,Fi,pi=move(ri,Fi,pi)
        if i%10==0:
            kin = kinetic(pi)
            energies_file(kin[0]+V_system(np.array(ri),R,f,L),kin[0],V_system(np.array(ri),R,f,L),kin[1],getP(get_funsum(ri)))
        if i%100==0:
            file2.write(str(N)+"\n\n")
            for pos in ri:
                file2.write("Ar"+"\t"+str(pos[0])+"\t"+str(pos[1])+"\t"+str(pos[2])+"\n")            
            file2.write("\n")
    file2.close()
    
movement(ri0,forces,pi0)