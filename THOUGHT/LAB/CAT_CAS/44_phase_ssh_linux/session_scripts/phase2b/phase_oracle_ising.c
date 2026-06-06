#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define NN 8
#define C 100
#define S 20

uint64_t lcg(uint64_t *s){*s=(*s*0x41C64E6D+0x3039);*s=(*s>>13)^*s;*s=(*s<<17)+*s;return *s;}
void decode(double *t,uint64_t *s){for(int i=0;i<NN;i++)s[i]=(cos(t[i])>=0)?1:0;}
int64_t ising(uint64_t *s,int64_t (*J)[NN]){int64_t E=0;for(int i=0;i<NN;i++){int si=(s[i]&1)?1:-1;for(int j=i+1;j<NN;j++){int sj=(s[j]&1)?1:-1;E-=J[i][j]*si*sj;}}return E;}
void brute(int64_t (*J)[NN],int64_t *g){int64_t b=INT64_MAX;for(uint64_t m=0;m<(1ULL<<NN);m++){uint64_t s[NN];for(int i=0;i<NN;i++)s[i]=(m>>i)&1;int64_t E=ising(s,J);if(E<b)b=E;}*g=b;}

// Jacobi eigenvalue decomposition for NxN symmetric matrix
// Returns eigenvectors in V (row-major: V[row*NN+col])
// Returns eigenvalues in eig
void jacobi(int64_t *Jflat,int NNloc,double *eig,double *V){
    double A[NN][NN];for(int i=0;i<NNloc;i++)for(int j=0;j<NNloc;j++)A[i][j]=Jflat[i*NNloc+j];
    // Init V as identity
    for(int i=0;i<NNloc;i++){for(int j=0;j<NNloc;j++)V[i*NNloc+j]=(i==j)?1.0:0.0;}
    for(int sweep=0;sweep<50;sweep++){
        for(int p=0;p<NNloc-1;p++)for(int q=p+1;q<NNloc;q++){
            double apq=A[p][q],app=A[p][p],aqq=A[q][q];
            if(fabs(apq)<1e-12)continue;
            double theta=(aqq-app)/(2*apq);
            double t=(theta>=0)?1.0/(theta+sqrt(1+theta*theta)):1.0/(theta-sqrt(1+theta*theta));
            double c=1.0/sqrt(1+t*t),s=c*t;
            // Rotate A
            A[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
            A[q][q]=s*s*app+2*c*s*apq+c*c*aqq;
            A[p][q]=A[q][p]=0;
            for(int r=0;r<NNloc;r++){
                if(r==p||r==q)continue;
                double arp=A[r][p],arq=A[r][q];
                A[r][p]=A[p][r]=c*arp-s*arq;
                A[r][q]=A[q][r]=s*arp+c*arq;
            }
            // Rotate V
            for(int r=0;r<NNloc;r++){
                double vrp=V[r*NNloc+p],vrq=V[r*NNloc+q];
                V[r*NNloc+p]=c*vrp-s*vrq;
                V[r*NNloc+q]=s*vrp+c*vrq;
            }
        }
    }
    for(int i=0;i<NNloc;i++)eig[i]=A[i][i];
    // simple sort by |eig| descending
    for(int i=0;i<NNloc-1;i++)for(int j=i+1;j<NNloc;j++)if(fabs(eig[i])<fabs(eig[j])){
        double t=eig[i];eig[i]=eig[j];eig[j]=t;
        for(int k=0;k<NNloc;k++){double tv=V[k*NNloc+i];V[k*NNloc+i]=V[k*NNloc+j];V[k*NNloc+j]=tv;}
    }
}

// Vertex-coordinate phase descent
void vertex_descend(double *theta,int64_t (*J)[NN],uint64_t *rng){
    for(int st=0;st<S;st++){int i=lcg(rng)%NN;double g=0;
        for(int j=0;j<NN;j++)if(J[i][j])g+=J[i][j]*sin(theta[i]-theta[j]);
        theta[i]-=0.1*g;}
}

// Spectral phase descent: theta = V * alpha, optimize alpha
void spectral_descend(double *alpha,double *V,double *eig,int64_t (*J)[NN],uint64_t *rng,int topK){
    double theta[NN];
    for(int st=0;st<S;st++){
        // Reconstruct theta
        for(int i=0;i<NN;i++){theta[i]=0;for(int k=0;k<topK;k++)theta[i]+=V[i*NN+k]*alpha[k];}
        int k=lcg(rng)%topK;
        // dE/dalpha_k = sum_i (dE/dtheta_i) * V[i][k]
        double g=0;
        for(int i=0;i<NN;i++){
            double dE_dt=0;
            for(int j=0;j<NN;j++)if(J[i][j])dE_dt+=J[i][j]*sin(theta[i]-theta[j]);
            g+=dE_dt*V[i*NN+k];
        }
        alpha[k]-=0.1*g;
    }
}

typedef struct{double best,mean,std,med;int hits;}st;
void stats(int64_t *E,int n,int64_t g,st *o){o->hits=0;o->best=1e9;double s=0;for(int i=0;i<n;i++){if(E[i]<o->best)o->best=E[i];if(E[i]==g)o->hits++;s+=E[i];}o->mean=s/n;int64_t t[n];memcpy(t,E,n*8);for(int i=0;i<n-1;i++)for(int j=i+1;j<n;j++)if(t[i]>t[j]){int64_t x=t[i];t[i]=t[j];t[j]=x;}o->med=n%2?t[n/2]:(t[n/2-1]+t[n/2])/2.0;double v=0;for(int i=0;i<n;i++)v+=(E[i]-o->mean)*(E[i]-o->mean);o->std=sqrt(v/n);}
void pst(const char*l,st*o){printf("  %-22s best=%3.0f mean=%+6.2f hits=%3d/%d\n",l,o->best,o->mean,o->hits,C);}

int main(){
    printf("=== PHASE 2B.5A v8: SPECTRAL PHASE ORACLE ===\n\n");
    printf("Jacobi eigenvalue decomposition + spectral phase coordinates.\n");
    printf("Compare: vertex oracle vs spectral oracle vs destructive nulls.\n\n");

    const char *nm[]={"Ferro","Anti-ferro","Mixed","RandSparse","FrustCycle","Planted"};
    int64_t Js[6][NN][NN]={{{0}}};int64_t gs[6];int ne[6];
    for(int i=0;i<NN-1;i++){Js[0][i][i+1]=Js[0][i+1][i]=1;}
    for(int i=0;i<NN-1;i++){Js[1][i][i+1]=Js[1][i+1][i]=-1;}
    for(int i=0;i<NN-1;i++){Js[2][i][i+1]=Js[2][i+1][i]=(i%2)?-1:1;}
    uint64_t rs=0xCADE000000000001ULL;
    for(int e=0;e<10;e++){int a=lcg(&rs)%NN,b=lcg(&rs)%NN;if(a!=b)Js[3][a][b]=Js[3][b][a]=(lcg(&rs)&1)?1:-1;}
    Js[4][0][1]=Js[4][1][0]=1;Js[4][1][2]=Js[4][2][1]=1;Js[4][0][2]=Js[4][2][0]=1;
    for(int i=2;i<NN-1;i++)Js[4][i][i+1]=Js[4][i+1][i]=1;
    int ss[NN]={1,0,1,0,1,0,1,0};for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)Js[5][i][j]=Js[5][j][i]=(ss[i]==ss[j])?1:-1;
    for(int p=0;p<6;p++){brute(Js[p],&gs[p]);ne[p]=0;for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)if(Js[p][i][j])ne[p]++;}

    for(int p=0;p<6;p++){
        printf("=== %s (%d edges, ground=%ld) ===\n",nm[p],ne[p],(long)gs[p]);

        // Jacobi decompose J as double
        int64_t Jflat[NN*NN];for(int i=0;i<NN;i++)for(int j=0;j<NN;j++)Jflat[i*NN+j]=Js[p][i][j];
        double eig[NN],V[NN*NN];jacobi(Jflat,NN,eig,V);
        printf("  Eigenvalues: ");for(int i=0;i<NN;i++)printf("%5.0f ",eig[i]);printf("\n");

        uint64_t rng=0x1111000000000001ULL+p*0x1000;
        int64_t E_vert[C],E_spec[C],E_rnd[C],E_act[C];
        int W_vs=0;

        for(int f=0;f<C;f++){
            double t0[NN];for(int i=0;i<NN;i++)t0[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);

            // Vertex oracle (v7 method)
            double tv[NN];memcpy(tv,t0,NN*8);vertex_descend(tv,Js[p],&rng);
            uint64_t sp[NN];decode(tv,sp);E_vert[f]=ising(sp,Js[p]);

            // Spectral oracle (v8 method): use top K modes
            double alpha[NN]={0};
            for(int k=0;k<NN;k++)alpha[k]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);
            spectral_descend(alpha,V,eig,Js[p],&rng,NN);
            double ts[NN];for(int i=0;i<NN;i++){ts[i]=0;for(int k=0;k<NN;k++)ts[i]+=V[i*NN+k]*alpha[k];}
            decode(ts,sp);E_spec[f]=ising(sp,Js[p]);
            if(E_spec[f]<E_vert[f])W_vs++;

            // Random null
            for(int i=0;i<NN;i++)sp[i]=lcg(&rng)&1;E_rnd[f]=ising(sp,Js[p]);

            // Active edge baseline
            for(int i=0;i<NN;i++)sp[i]=lcg(&rng)&1;
            for(int it=0;it<5000;it++){int imp=0;for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)if(Js[p][i][j]){int sg=Js[p][i][j];if(sg>0&&(sp[i]&1)!=(sp[j]&1)){sp[i]^=1;imp=1;}else if(sg<0&&(sp[i]&1)==(sp[j]&1)){sp[i]^=1;imp=1;}}if(!imp)break;}
            E_act[f]=ising(sp,Js[p]);
        }

        st sv,sspec,sr,sa;
        stats(E_vert,C,gs[p],&sv);stats(E_spec,C,gs[p],&sspec);stats(E_rnd,C,gs[p],&sr);stats(E_act,C,gs[p],&sa);
        pst("Vertex oracle", &sv);
        pst("Spectral oracle", &sspec);
        pst("Random null", &sr);
        pst("Active edge", &sa);
        printf("  Spectral vs Vertex: mean=%+.2f wins=%d/%d\n\n",sspec.mean-sv.mean,W_vs,C);
    }

    printf("=== VERDICT ===\n");
    printf("Jacobi eigen decomposition + spectral phase coordinates implemented.\n");
    printf("Vertex (theta-space) vs spectral (alpha-space via V*alpha) comparison.\n");
    printf("Spectral oracle uses graph-native eigenbasis for phase optimization.\n");
    printf("PHASE2B_5A_V8_SPECTRAL_PHASE_ORACLE_COMPLETE\n");
    return 0;
}
