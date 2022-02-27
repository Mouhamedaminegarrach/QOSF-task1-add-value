from qiskit import  QuantumCircuit, Aer, execute,BasicAer
from qiskit.visualization import plot_histogram
import Functions as oq
import numpy as np
import math as m
S_simulator = Aer.backends(name='statevector_simulator')[0]

################################################################
def QRAM():
    
    #give a superposition with hadamard gate with 3 qubits
    #(n=5)+(m=5)+3=13
    qc = QuantumCircuit(13,13)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.barrier()

    #create the first part (position for each number in the list using NOT and multi-controled X gate) n=5
    qc.mcx([0,1,2],3)
    qc.x(2)
    qc.mcx([0,1,2],4)
    qc.x(2)
    qc.x(1)
    qc.mcx([0,1,2],5)
    qc.x(2)
    qc.mcx([0,1,2],6)
    qc.x(2)
    qc.x(1)
    qc.x(0)
    qc.mcx([0,1,2],7)
    qc.x(0)
    qc.barrier()

    #Second part (Value for each number in the list in banery) m=5
    qc.cx(3,10)
    qc.cx(3,12)
    qc.cx(4,10)
    qc.cx(4,11)
    qc.cx(4,12)
    qc.cx(5,9)
    qc.cx(6,9)
    qc.cx(6,12)
    qc.cx(7,12)
    qc.barrier()

    qc.measure([0,1,2,3,4,5,6,7,8,9,10,11,12],[0,1,2,3,4,5,6,7,8,9,10,11,12])
    qc.draw(output='mpl') #draw our QRAM circuit

    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend)
    plot_histogram(job.result().get_counts()) 
    

    counts = job.result().get_counts()
    print(qc)
    
    return(counts)
################################################################

def Binary(N, total, LSB):
    '''
    Input: N (integer) total (integer) LSB (string)
    Returns the base-2 binary equivilant of N according to left or right least significant bit notation
    '''
    qubits = int(m.log(total,2))
    b_num = np.zeros(qubits)
    for i in np.arange(qubits):
        if( N/((2)**(qubits-i-1)) >= 1 ):
            if(LSB=='R'):
                b_num[i] = 1
            if(LSB=='L'):
                b_num[int(qubits-(i+1))] = 1
                N = N - 2**(qubits-i-1)
    B = []
    for j in np.arange(len(b_num)):
        B.append(int(b_num[j]))
    return B

###############################################################

def QFT(qc, q, qubits, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) qubits (integer)
    Keyword Arguments: swap (Bool) - Adds SWAP gates after all of the phase gates have been applied
    Assigns all the gate operations for a Quantum Fourier Transformation'''
    R_phis = [0]
    for i in np.arange(2,int(qubits+1)):
        R_phis.append( 2/(2**(i)) * m.pi )
    for j in np.arange(int(qubits)):
        qc.h( q[int(j)] )
        for k in np.arange(int(qubits-(j+1))):
            qc.cu1( R_phis[k+1], q[int(j+k+1)], q[int(j)] )
    if 'swap' in kwargs:
        if(kwargs['swap'] == True):
            for s in np.arange(m.floor(qubits/2.0)):
                qc.swap( q[int(s)],q[int(qubits-1-s)] )

################################################################
def QFT_dgr(qc, q, qubits, **kwargs):
    '''
    Input: qc (QuantumCircuit) q (QuantumRegister) qubits (integer)
    Keyword Arguments: swap (Bool) - Adds SWAP gates after all of the phase gates have been applied
    Assigns all the gate operations for a Quantum Fourier Transformation
    '''
    if 'swap' in kwargs:
        if(kwargs['swap'] == True):
            for s in np.arange(m.floor(qubits/2.0)):
                qc.swap( q[int(s)],q[int(qubits-1-s)] )
    R_phis = [0]
    for i in np.arange(2,int(qubits+1)):
        R_phis.append( -2/(2**(i)) * m.pi )
    for j in np.arange(int(qubits)):
        for k in np.arange(int(j)):
            qc.cu1(R_phis[int(j-k)], q[int(qubits-(k+1))], q[int(qubits-(j+1))] )
            qc.h( q[int(qubits-(j+1))] )

################################################################
def Wavefunction(obj, **kwargs):
    '''
    Prints a tidier versrion of the array statevector corresponding to the wavefuntion of a QuantumCircuit object
    Keyword Arguments: precision (integer) - the decimal precision for amplitudes
    column (Bool) - prints each state in a vertical column
    systems (array of integers) - seperates the qubits into different states
    show_systems (array of Bools) - indictates which qubit systems to print
    '''
    if(type(obj) == QuantumCircuit ):
        statevec = execute( obj, S_simulator, shots=1 ).result().get_statevector()
    if(type(obj) == np.ndarray):
        statevec = obj
    sys = False
    NL = False
    dec = 5
    if 'precision' in kwargs:
        dec = int( kwargs['precision'] )
    if 'column' in kwargs:
        NL = kwargs['column']
    if 'systems' in kwargs:
        systems = kwargs['systems']
        sys = True
        last_sys = int(len(systems)-1)
        show_systems = []
        for s_chk in np.arange(len(systems)):
            if( type(systems[s_chk])!=int ):
                raise Exception('systems must be an array of all integers')
        if 'show_systems' in kwargs:
            show_systems = kwargs['show_systems']
            if( len(systems)!=len(show_systems) ):
                raise Exception('systems and show_systems need to be arrays of equal length')
            for ls in np.arange(len(show_systems)):
                if((show_systems[ls]!=True)and(show_systems[ls]!=False)):
                    raise Exception('show_systems must be an array of Truth Values')
                if(show_systems[ls]==True):
                    last_sys = int(ls)
        else:
            for ss in np.arange(len(systems)):
                show_systems.append(True)
    wavefunction = []
    qubits = int(m.log(len(statevec),2))
    for i in np.arange( int(len(statevec)) ):
        value = round(statevec[i].real, dec) + round(statevec[i].imag, dec) * 1j
        if( (value.real!=0) or (value.imag!=0) ):
            state = list(Binary(int(i),int(2**qubits),'L'))
            state_str = ''
            if( sys == True ):
                k = 0
                for s in np.arange(len(systems)):
                    if(show_systems[s]==True):
                        if(int(s)!=last_sys):
                            state.insert( int(k+systems[s]),'>|' )
                            k = int(k+systems[s]+1)
                        else:
                            k = int(k+systems[s])
                    else:
                        for s2 in np.arange(systems[s]):
                            del state[int(k)]

            for j in np.arange(len(state)):
                if(type(state[j])!=str):
                    state_str = state_str+str(int(state[j]))
                else:
                    state_str = state_str+state[j]
            if( (value.real!=0) and (value.imag!=0) ):
                if( value.imag > 0):
                    wavefunction = wavefunction + [state_str]
                else:
                    wavefunction = wavefunction +[state_str]
            if( (value.real!=0) and (value.imag==0) ):
                wavefunction = wavefunction +[state_str]
            if( (value.real==0) and (value.imag!=0) ):
                wavefunction = wavefunction +[state_str]
            

    return(wavefunction)

################################################################
def Quantum_Adder(qc, Qa, Qb, A, B):
    '''
    Input: qc (QuantumCircuit) Qa (QuantumRegister) Qb (QuantumRegister) A (array) B (array)
    Appends all of the gate operations for a QFT based addition of two states A and B
    '''  
    Q = len(B)
    for n in np.arange(Q):
        if( A[n] == 1 ):
            qc.x( Qa[int(n+1)] )
        if( B[n] == 1 ):
            qc.x( Qb[int(n)] )
    QFT(qc,Qa,Q+1)
    p = 1
    for j in np.arange( Q ):
        qc.cu1( m.pi/(2**p), Qb[int(j)], Qa[0] )
        p += 1
    for i in np.arange(1,Q+1):
        p = 0
        for jj in np.arange( i-1, Q ):
            qc.cu1( m.pi/(2**p), Qb[int(jj)], Qa[int(i)] )
            p =p+ 1
    QFT_dgr(qc,Qa,Q+1)
