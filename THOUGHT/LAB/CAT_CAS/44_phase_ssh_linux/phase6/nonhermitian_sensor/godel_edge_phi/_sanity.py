import sys, os, numpy as np
HERE=os.path.dirname(os.path.abspath('.'))
FOLD=r"D:/CCC 2.0/AI/agent-governance-system/THOUGHT/LAB/CAT_CAS/44_phase_ssh_linux/phase6/fold_audit"
sys.path.insert(0,FOLD); sys.path.insert(0,os.getcwd())
import construction as C
import no_smuggle_gate as G
import godel_operator as O

rng=np.random.default_rng(7)
n=8; N=1<<n
d=C.sample_secret(N,rng)
inst=G.make_instance(n,d,rng)
print("n=%d N=%d d=%d  orient=%d  a=min(d,N-d)=%d"%(n,N,d,C.orientation_bit(d,N),min(d,N-d)))

# accept hits exactly the two fixed points?
acc=O.accept_profile(inst["k"],inst["b"],N)
hits=np.where(acc>0)[0]
print("accepted sites:", hits, " expect {%d,%d}"%(d,N-d))

# closed-form vs numeric winding
diag=(-1j*O.ELL)+O.S_LOOP*acc
feat, Wnum, Wclosed = O.phi_features(diag, O.E_REFS, want_numeric=True)
print("W closed:", Wclosed, " W numeric:", Wnum, " match:", bool(np.all(Wnum==Wclosed)))

# fold invariance of public readout (delta must be 0)
f0=O.O_public_godel_phi(inst)
f1=O.O_public_godel_phi(G.folded_instance(inst))
print("PUBLIC fold-delta:", float(np.max(np.abs(f0-f1))))
print("feature dim:", f0.shape[0])

# smuggle readout must FLIP under the fold
s0=O.O_smuggle_godel_phi(inst)
s1=O.O_smuggle_godel_phi(G.folded_instance(inst))
print("SMUGGLE fold-delta:", float(np.max(np.abs(s0-s1))))

# direct dense determinant cross-check of the rank-1 closed form at small N
n2=6; N2=1<<n2
d2=C.sample_secret(N2,rng); inst2=G.make_instance(n2,d2,rng)
acc2=O.accept_profile(inst2["k"],inst2["b"],N2)
diag2=(-1j*O.ELL)+O.S_LOOP*acc2
tR=np.exp(O.A_HOP); tL=np.exp(-O.A_HOP)
# build dense open T
T=np.zeros((N2,N2),dtype=complex)
for x in range(N2):
    T[x,(x-1)%N2]=0; T[x,(x+1)%N2]=0
for x in range(1,N2): T[x,x-1]=tR
for x in range(0,N2-1): T[x,x+1]=tL
np.fill_diagonal(T,diag2)
R=O.LOOP_RADIUS; lam=R*np.exp(-O.A_HOP*(N2-1))
Eref=0.3+0.2j
import numpy.linalg as la
# dense det of (E I - H(phi)) with H[0,N2-1]=lam e^{iphi}
for phi in (0.0,1.3,2.9):
    H=T.copy(); H[0,N2-1]=lam*np.exp(1j*phi)
    dense=la.det(Eref*np.eye(N2)-H)
    lm,ph=O.tridet_log_vec(diag2,np.array([Eref]),1.0)
    Dopen=np.exp(lm[0])*ph[0]
    closed=Dopen - lam*(tR**(N2-1))*np.exp(1j*phi)
    print("phi=%.2f dense=%.4e%+.4ej closed=%.4e%+.4ej relerr=%.2e"%(
        phi, dense.real,dense.imag, closed.real,closed.imag, abs(dense-closed)/max(abs(dense),1e-12)))
