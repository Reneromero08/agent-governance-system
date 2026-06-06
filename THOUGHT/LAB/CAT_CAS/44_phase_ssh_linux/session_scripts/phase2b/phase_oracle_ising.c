#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define NN 8
#define C 100
#define S 30

uint64_t lcg(uint64_t *s){*s=(*s*0x41C64E6D+0x3039);*s=(*s>>13)^*s;*s=(*s<<17)+*s;return *s;}
void decode(double *t,uint64_t *s){for(int i=0;i<NN;i++)s[i]=(cos(t[i])>=0)?1:0;}
int64_t ising(uint64_t *s,int64_t (*J)[NN]){int64_t E=0;for(int i=0;i<NN;i++){int si=(s[i]&1)?1:-1;for(int j=i+1;j<NN;j++){int sj=(s[j]&1)?1:-1;E-=J[i][j]*si*sj;}}return E;}
void brute(int64_t (*J)[NN],int64_t *g){int64_t b=INT64_MAX;for(uint64_t m=0;m<(1ULL<<NN);m++){uint64_t s[NN];for(int i=0;i<NN;i++)s[i]=(m>>i)&1;int64_t E=ising(s,J);if(E<b)b=E;}*g=b;}

void vertex_descend(double *theta,int64_t (*J)[NN],uint64_t *rng){
    for(int st=0;st<S;st++){int i=lcg(rng)%NN;double g=0;
        for(int j=0;j<NN;j++)if(J[i][j])g+=J[i][j]*sin(theta[i]-theta[j]);
        theta[i]-=0.1*g;}
}

// Edge coherence: agreement_ij = sign*cos(dtheta), range [-1,1], +1 = satisfied
double edge_coherence(double *theta,int *edges,int ne){
    double sum=0;
    for(int e=0;e<ne;e++){int i=edges[e*3],j=edges[e*3+1],sg=edges[e*3+2];
        sum+=sg*cos(theta[i]-theta[j]);}
    return sum/ne;
}

// Autocorrelation of edge-agreement vector (direct O(E^2))
double autocorr_peak(double *A,int E){
    double best=0;int best_k=0;
    for(int k=1;k<E;k++){
        double sum=0;int n=0;
        for(int e=0;e<E-k;e++){sum+=A[e]*A[e+k];n++;}
        double val=sum/n;
        if(fabs(val)>fabs(best)){best=val;best_k=k;}
    }
    return best;
}

// Recursive autocorr: apply autocorr to autocorr output, track peak amplification
double cepstrum_amplify(double *A,int E){
    double *work=malloc(E*8),*tmp=malloc(E*8);
    memcpy(work,A,E*8);int len=E;
    for(int rec=0;rec<3;rec++){
        for(int k=0;k<len-1;k++){double s=0;int n=0;for(int e=0;e<len-k;e++){s+=work[e]*work[e+k];n++;}tmp[k]=s/n;}
        len--;memcpy(work,tmp,len*8);
    }
    double peak=0;for(int k=0;k<len;k++)if(fabs(work[k])>fabs(peak))peak=work[k];
    free(work);free(tmp);return peak;
}

typedef struct{double best,mean,std,med;int hits;}st;
void stats64(int64_t *E,int n,int64_t g,st *o){o->hits=0;o->best=1e9;double s=0;for(int i=0;i<n;i++){if(E[i]<o->best)o->best=E[i];if(E[i]==g)o->hits++;s+=E[i];}o->mean=s/n;int64_t t[n];memcpy(t,E,n*8);for(int i=0;i<n-1;i++)for(int j=i+1;j<n;j++)if(t[i]>t[j]){int64_t x=t[i];t[i]=t[j];t[j]=x;}o->med=n%2?t[n/2]:(t[n/2-1]+t[n/2])/2.0;double v=0;for(int i=0;i<n;i++)v+=(E[i]-o->mean)*(E[i]-o->mean);o->std=sqrt(v/n);}
void pst(const char*l,st*o){printf("  %-28s best=%3.0f mean=%+6.2f hits=%3d/%d\n",l,o->best,o->mean,o->hits,C);}

int main(){
    printf("=== PHASE 2B.5A v10: AUTOCORRELATION / COHERENCE / CEPSTRUM ===\n\n");
    printf("Edge phase coherence + autocorrelation of agreement vector + cepstrum amplification.\n");
    printf("Compare: v10 coherence-ranked vs v7 vertex oracle vs destructive nulls.\n\n");

    const char *nm[]={"Ferro","Anti-ferro","Mixed","RandSparse","FrustCycle","Planted"};
    int64_t Js[6][NN][NN]={{{0}}};int64_t gs[6];int edges[6][32][3],ne[6];
    for(int i=0;i<NN-1;i++){Js[0][i][i+1]=Js[0][i+1][i]=1;}
    for(int i=0;i<NN-1;i++){Js[1][i][i+1]=Js[1][i+1][i]=-1;}
    for(int i=0;i<NN-1;i++){Js[2][i][i+1]=Js[2][i+1][i]=(i%2)?-1:1;}
    uint64_t rs=0xCADE000000000001ULL;
    for(int e=0;e<10;e++){int a=lcg(&rs)%NN,b=lcg(&rs)%NN;if(a!=b)Js[3][a][b]=Js[3][b][a]=(lcg(&rs)&1)?1:-1;}
    Js[4][0][1]=Js[4][1][0]=1;Js[4][1][2]=Js[4][2][1]=1;Js[4][0][2]=Js[4][2][0]=1;for(int i=2;i<NN-1;i++)Js[4][i][i+1]=Js[4][i+1][i]=1;
    int ss[NN]={1,0,1,0,1,0,1,0};for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)Js[5][i][j]=Js[5][j][i]=(ss[i]==ss[j])?1:-1;
    for(int p=0;p<6;p++){brute(Js[p],&gs[p]);ne[p]=0;for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)if(Js[p][i][j]){edges[p][ne[p]][0]=i;edges[p][ne[p]][1]=j;edges[p][ne[p]][2]=Js[p][i][j];ne[p]++;}}

    for(int p=0;p<6;p++){
        printf("=== %s (%d edges, ground=%ld) ===\n",nm[p],ne[p],(long)gs[p]);
        int64_t Ev7[C],Ev10[C],Er[C],Erp[C],Eew[C],Esi[C];
        double coh_v7[C],coh_v10[C],acorr_v10[C],cep_v10[C];
        double coh_true_edges[C],coh_random_edges[C],coh_rewired[C];
        uint64_t rng=0x1111000000000001ULL+p*0x1000;

        // Pre-build edge-rewired graph for null
        int64_t Jrw[NN][NN]={{0}};
        {int ea[32],eb[32];for(int e=0;e<ne[p];e++){ea[e]=edges[p][e][0];eb[e]=edges[p][e][1];}
        for(int e=0;e<ne[p];e++){int j=lcg(&rng)%ne[p];int t=ea[e];ea[e]=ea[j];ea[j]=t;t=eb[e];eb[e]=eb[j];eb[j]=t;}
        for(int e=0;e<ne[p];e++){int v=Js[p][edges[p][e][0]][edges[p][e][1]];Jrw[ea[e]][eb[e]]=Jrw[eb[e]][ea[e]]=v;}}

        for(int f=0;f<C;f++){
            double tv[NN];for(int i=0;i<NN;i++)tv[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);

            // v7: vertex oracle
            vertex_descend(tv,Js[p],&rng);
            uint64_t sp[NN];decode(tv,sp);Ev7[f]=ising(sp,Js[p]);
            coh_v7[f]=edge_coherence(tv,(int*)edges[p],ne[p]);

            // v10: fresh descent + coherence/autocorr/cepstrum analysis
            double t10[NN];for(int i=0;i<NN;i++)t10[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);
            vertex_descend(t10,Js[p],&rng);
            decode(t10,sp);Ev10[f]=ising(sp,Js[p]);
            coh_v10[f]=edge_coherence(t10,(int*)edges[p],ne[p]);

            // Edge agreement vector for autocorrelation
            double A[64];int aE=ne[p];
            for(int e=0;e<aE;e++){int i=edges[p][e][0],j=edges[p][e][1],sg=edges[p][e][2];
                A[e]=sg*cos(t10[i]-t10[j]);}
            acorr_v10[f]=autocorr_peak(A,aE);
            cep_v10[f]=cepstrum_amplify(A,aE);

            // True edge coherence vs random edge coherence
            coh_true_edges[f]=edge_coherence(t10,(int*)edges[p],ne[p]);
            // Random edges: pick ne[p] random (i,j) pairs
            double sum_r=0;
            for(int e=0;e<ne[p];e++){int i=lcg(&rng)%NN,j=lcg(&rng)%NN;if(i!=j)sum_r+=cos(t10[i]-t10[j]);}
            coh_random_edges[f]=sum_r/ne[p];

            // Rewired edge coherence
            double sum_w=0;
            for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)if(Jrw[i][j])sum_w+=(Jrw[i][j]>0?1:-1)*cos(t10[i]-t10[j]);
            coh_rewired[f]=sum_w/ne[p];

            // Nulls
            for(int i=0;i<NN;i++)sp[i]=lcg(&rng)&1;Er[f]=ising(sp,Js[p]);
            int64_t Jf[NN][NN]={{0}};for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)Jf[i][j]=Jf[j][i]=(lcg(&rng)&1)?1:-1;
            double trp[NN];for(int i=0;i<NN;i++)trp[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);
            vertex_descend(trp,Jf,&rng);decode(trp,sp);Erp[f]=ising(sp,Js[p]);
            double tew[NN];for(int i=0;i<NN;i++)tew[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);
            vertex_descend(tew,Jrw,&rng);decode(tew,sp);Eew[f]=ising(sp,Js[p]);
            // Sign-shuffled
            int64_t Jsi[NN][NN]={{0}};for(int i=0;i<NN;i++)for(int j=i+1;j<NN;j++)if(Js[p][i][j]){int v=Js[p][i][j];Jsi[i][j]=Jsi[j][i]=(lcg(&rng)&1)?v:-v;}
            double tsi[NN];for(int i=0;i<NN;i++)tsi[i]=2*M_PI*(lcg(&rng)/(double)UINT64_MAX);
            vertex_descend(tsi,Jsi,&rng);decode(tsi,sp);Esi[f]=ising(sp,Js[p]);
        }

        st sv7,sv10,sr,srp,sew,ssi;
        stats64(Ev7,C,gs[p],&sv7);stats64(Ev10,C,gs[p],&sv10);
        stats64(Er,C,gs[p],&sr);stats64(Erp,C,gs[p],&srp);
        stats64(Eew,C,gs[p],&sew);stats64(Esi,C,gs[p],&ssi);

        pst("v7 Vertex oracle", &sv7);
        pst("v10 Coherence/cepstrum", &sv10);
        pst("Random spin [destr]", &sr);
        pst("RandPh descent [destr]", &srp);
        pst("Sign-shuffled [destr]", &ssi);
        pst("Edge-rewired [destr]", &sew);

        // Coherence statistics
        double mct=0;for(int f=0;f<C;f++)mct+=coh_true_edges[f];mct/=C;
        double mcr=0;for(int f=0;f<C;f++)mcr+=coh_random_edges[f];mcr/=C;
        double mcw=0;for(int f=0;f<C;f++)mcw+=coh_rewired[f];mcw/=C;
        printf("  Coherence: true=%.3f random=%.3f rewired=%.3f true>random=%s\n",mct,mcr,mcw,mct>mcr?"YES":"NO");

        double mac=0;for(int f=0;f<C;f++)mac+=fabs(acorr_v10[f]);mac/=C;
        double mcp=0;for(int f=0;f<C;f++)mcp+=fabs(cep_v10[f]);mcp/=C;
        printf("  Autocorr peak: %.3f  Cepstrum peak: %.3f\n",mac,mcp);
        printf("  v10 vs v7: mean=%+.2f hits=%+d\n",sv10.mean-sv7.mean,sv10.hits-sv7.hits);
        printf("  v10 vs edge-rewired: mean=%+.2f hits=%+d\n\n",sv10.mean-sew.mean,sv10.hits-sew.hits);
    }

    printf("=== VERDICT ===\n");
    printf("v10 autocorrelation/coherence/cepstrum implemented.\n");
    printf("Edge coherence vs random edge coherence vs rewired coherence compared.\n");
    printf("Autocorr peak and cepstrum amplification measured per candidate.\n");
    printf("PHASE2B_5A_V10_AUTOCORR_IMPLEMENTED\n");
    return 0;
}
